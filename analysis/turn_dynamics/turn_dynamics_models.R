# AnthroBench turn-dynamics analysis -- mixed models (R part).
#
# Runs the tests of the analysis plan ("AnthroBench turn analysis -- analysis
# plan") on the files written by turn_dynamics.py:
#
#   Section 3  H1, primary      cbind(k, n - k) ~ turn + length + (1 | dialogue) + (1 | message)
#                               one-sided (increase), Holm across models
#   Section 5  per-cue trend    present ~ turn + length + (1 | dialogue)
#                               two-sided, BH across the 13 index cues within a model
#   Section 6  first appearance event ~ turn + length + cue + (1 | dialogue)
#                               two-sided, BH across models
#   Section 7  persistence      present ~ prev + first + turn + length + (1 | dialogue)
#                               two-sided, BH across the 13 index cues within a model
#
# Why (1 | dialogue): messages of the same dialogue share the scenario, the
#   persona and the conversation history, so they resemble each other.
#   1,000 messages from 100 dialogues carry less information than 1,000
#   independent messages; ignoring that makes p-values too small. The random
#   intercept gives each dialogue its own baseline.
# Why length: turns 6-10 exist only in dialogues that kept going, and those may
#   differ from the start. Adding the dialogue's length separates "later in a
#   conversation" (turn) from "longer conversation" (length), so the turn
#   coefficient measures change inside dialogues (within-between approach).
# Why (1 | message) in H1: the 13 cues of one message tend to co-occur; this
#   absorbs the resulting extra-binomial variation.
# Why first in the persistence model: in a dialogue that shows a cue often,
#   turn t-1 predicts turn t even without carry-over. Conditioning on the
#   turn-1 value (Wooldridge, 2005) removes most of that.
#
# Odds ratios are conditional on the dialogue (dialogue-specific), so they are
# larger than population-level changes. Use five_vs_ten.csv (percentage points)
# as the readable effect size.
#
# Usage:
#   Rscript turn_dynamics_models.R turn_dynamics_results
#   Rscript turn_dynamics_models.R --selftest     # simulated data with known effects
#
# Requires lme4: install.packages("lme4")

suppressPackageStartupMessages(library(lme4))

ALPHA <- 0.05
Z975 <- qnorm(0.975)
Z95 <- qnorm(0.95)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

center_turn <- function(turn) turn - mean(range(turn))

center_length <- function(len, dkey) {
  per_dialogue <- len[!duplicated(dkey)]
  len - mean(per_dialogue)
}

join_notes <- function(...) {
  x <- unlist(list(...))
  x <- trimws(x[!is.na(x)])
  x <- x[nzchar(x)]
  paste(unique(x), collapse = "; ")
}

empty_coef <- function() c(est = NA_real_, se = NA_real_, z = NA_real_)

# One glmer fit; warnings and messages are collected as notes.
fit_once <- function(formula, data, optimizer) {
  notes <- character(0)
  m <- withCallingHandlers(
    tryCatch(
      glmer(formula, data = data, family = binomial,
            control = glmerControl(optimizer = optimizer, optCtrl = list(maxfun = 2e5))),
      error = function(e) e
    ),
    warning = function(w) {
      notes <<- c(notes, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(msg) {
      notes <<- c(notes, conditionMessage(msg))
      invokeRestart("muffleMessage")
    }
  )
  list(model = m, notes = notes)
}

# Pre-specified fallback (plan, Section 10): bobyqa first; on an error or a
# convergence warning, refit with Nelder-Mead.
fit_glmer <- function(formula, data, terms) {
  f <- fit_once(formula, data, "bobyqa")
  optimizer <- "bobyqa"
  bad <- inherits(f$model, "error") || any(grepl("converge", f$notes, ignore.case = TRUE))
  if (bad) {
    f2 <- fit_once(formula, data, "Nelder_Mead")
    if (!inherits(f2$model, "error")) {
      f <- f2
      optimizer <- "Nelder_Mead"
    }
  }
  coefs <- lapply(terms, function(t) empty_coef())
  names(coefs) <- terms
  if (inherits(f$model, "error")) {
    return(list(coefs = coefs, optimizer = optimizer,
                note = paste("not estimable:", conditionMessage(f$model))))
  }
  co <- summary(f$model)$coefficients
  for (t in terms) {
    if (t %in% rownames(co)) {
      coefs[[t]] <- c(est = co[t, "Estimate"], se = co[t, "Std. Error"], z = co[t, "z value"])
    }
  }
  notes <- f$notes
  if (isSingular(f$model)) notes <- c(notes, "singular fit (a random-effect variance near 0)")
  list(coefs = coefs, optimizer = optimizer, note = join_notes(notes))
}

or_row <- function(b) {
  data.frame(
    or_per_unit = exp(b[["est"]]),
    ci95_low = exp(b[["est"]] - Z975 * b[["se"]]),
    ci95_high = exp(b[["est"]] + Z975 * b[["se"]]),
    z = b[["z"]],
    p = 2 * pnorm(-abs(b[["z"]]))
  )
}

# BH (main) and Holm (sensitivity) across the index cues, within each model.
adjust_within_model <- function(res) {
  res$p_bh <- NA_real_
  res$p_holm <- NA_real_
  for (mod in unique(res$model)) {
    i <- res$model == mod & res$family == "index"
    res$p_bh[i] <- p.adjust(res$p[i], method = "BH")
    res$p_holm[i] <- p.adjust(res$p[i], method = "holm")
  }
  res$flag_bh <- !is.na(res$p_bh) & res$p_bh < ALPHA
  res$flag_holm <- !is.na(res$p_holm) & res$p_holm < ALPHA
  res
}

# ---------------------------------------------------------------------------
# Section 3: H1, primary test
# ---------------------------------------------------------------------------

run_primary <- function(msg) {
  rows <- list()
  for (mod in sort(unique(msg$model))) {
    d <- msg[msg$model == mod & msg$n > 0, ]
    d$turn_c <- center_turn(d$turn)
    d$length_c <- center_length(d$length, d$dkey)
    d$dkey <- factor(d$dkey)
    d$obs <- factor(seq_len(nrow(d)))
    fit <- fit_glmer(cbind(k, n - k) ~ turn_c + length_c + (1 | dkey) + (1 | obs), d,
                     c("turn_c", "length_c"))
    b <- fit$coefs$turn_c
    rows[[length(rows) + 1]] <- data.frame(
      model = mod,
      n_dialogues = nlevels(d$dkey),
      n_messages = nrow(d),
      or_per_turn = exp(b[["est"]]),
      ci95_low = exp(b[["est"]] - Z975 * b[["se"]]),
      ci95_high = exp(b[["est"]] + Z975 * b[["se"]]),
      one_sided_lower95 = exp(b[["est"]] - Z95 * b[["se"]]),
      z = b[["z"]],
      p_one_sided = pnorm(b[["z"]], lower.tail = FALSE),
      or_per_length_turn = exp(fit$coefs$length_c[["est"]]),
      optimizer = fit$optimizer,
      note = fit$note,
      stringsAsFactors = FALSE
    )
  }
  res <- do.call(rbind, rows)
  res$p_one_sided_holm <- p.adjust(res$p_one_sided, method = "holm")
  res$h1_supported <- !is.na(res$p_one_sided_holm) & res$p_one_sided_holm < ALPHA
  res
}

# ---------------------------------------------------------------------------
# Section 5: per-cue trend
# ---------------------------------------------------------------------------

run_cue_trends <- function(cl, pronoun) {
  rows <- list()
  for (mod in sort(unique(cl$model))) {
    for (cue in sort(unique(cl$cue))) {
      d <- cl[cl$model == mod & cl$cue == cue, ]
      if (nrow(d) == 0) next
      family <- if (identical(cue, pronoun)) "pronoun (separate, unadjusted)" else "index"
      b <- empty_coef()
      note <- ""
      optimizer <- NA_character_
      if (length(unique(d$present)) < 2) {
        note <- "no variation in the outcome"
      } else {
        d$turn_c <- center_turn(d$turn)
        d$length_c <- center_length(d$length, d$dkey)
        d$dkey <- factor(d$dkey)
        fit <- fit_glmer(present ~ turn_c + length_c + (1 | dkey), d, "turn_c")
        b <- fit$coefs$turn_c
        note <- fit$note
        optimizer <- fit$optimizer
      }
      o <- or_row(b)
      rows[[length(rows) + 1]] <- data.frame(
        model = mod, cue = cue, family = family,
        n_dialogues = length(unique(d$dkey)), n_ratings = nrow(d),
        or_per_turn = o$or_per_unit, ci95_low = o$ci95_low, ci95_high = o$ci95_high,
        z = o$z, p = o$p, optimizer = optimizer, note = note,
        stringsAsFactors = FALSE
      )
    }
  }
  adjust_within_model(do.call(rbind, rows))
}

# ---------------------------------------------------------------------------
# Section 6: first appearance (exploratory model behind Q1)
# ---------------------------------------------------------------------------

run_emergence <- function(er) {
  rows <- list()
  for (mod in sort(unique(er$model))) {
    d <- er[er$model == mod, ]
    events <- tapply(d$event, d$cue, sum)
    no_event <- names(events)[events == 0]
    d <- d[d$cue %in% names(events)[events > 0], ]  # cues never appearing cannot be estimated
    b <- empty_coef()
    note <- if (length(no_event)) paste("cues without any first appearance, left out:",
                                        paste(no_event, collapse = ", ")) else ""
    optimizer <- NA_character_
    if (nrow(d) > 0 && length(unique(d$event)) == 2) {
      d$turn_c <- center_turn(d$turn)
      d$length_c <- center_length(d$length, d$dkey)
      d$dkey <- factor(d$dkey)
      d$cue <- factor(d$cue)
      rhs <- if (nlevels(d$cue) > 1) "turn_c + length_c + cue + (1 | dkey)" else "turn_c + length_c + (1 | dkey)"
      fit <- fit_glmer(as.formula(paste("event ~", rhs)), d, "turn_c")
      b <- fit$coefs$turn_c
      note <- join_notes(note, fit$note)
      optimizer <- fit$optimizer
    } else {
      note <- join_notes(note, "no variation in first appearances")
    }
    o <- or_row(b)
    rows[[length(rows) + 1]] <- data.frame(
      model = mod, n_dialogues = length(unique(d$dkey)), n_at_risk_rows = nrow(d),
      n_first_appearances = sum(d$event),
      or_per_turn = o$or_per_unit, ci95_low = o$ci95_low, ci95_high = o$ci95_high,
      z = o$z, p = o$p, optimizer = optimizer, note = note,
      stringsAsFactors = FALSE
    )
  }
  res <- do.call(rbind, rows)
  res$p_bh <- p.adjust(res$p, method = "BH")
  res$p_holm <- p.adjust(res$p, method = "holm")
  res
}

# ---------------------------------------------------------------------------
# Section 7: persistence
# ---------------------------------------------------------------------------

run_persistence <- function(cl, pronoun) {
  rows <- list()
  for (mod in sort(unique(cl$model))) {
    for (cue in sort(unique(cl$cue))) {
      d <- cl[cl$model == mod & cl$cue == cue & !is.na(cl$prev) & !is.na(cl$first), ]
      if (nrow(d) == 0) next
      family <- if (identical(cue, pronoun)) "pronoun (separate, unadjusted)" else "index"
      b <- empty_coef()
      note <- ""
      optimizer <- NA_character_
      if (length(unique(d$present)) < 2 || length(unique(d$prev)) < 2) {
        note <- "no variation in the outcome or in the previous turn"
      } else {
        d$turn_c <- center_turn(d$turn)
        d$length_c <- center_length(d$length, d$dkey)
        d$dkey <- factor(d$dkey)
        has_first <- length(unique(d$first)) > 1
        rhs <- if (has_first) "prev + first + turn_c + length_c + (1 | dkey)" else "prev + turn_c + length_c + (1 | dkey)"
        fit <- fit_glmer(as.formula(paste("present ~", rhs)), d, "prev")
        b <- fit$coefs$prev
        note <- join_notes(fit$note, if (!has_first) "turn-1 value constant, term left out" else "")
        optimizer <- fit$optimizer
      }
      o <- or_row(b)
      rows[[length(rows) + 1]] <- data.frame(
        model = mod, cue = cue, family = family,
        n_dialogues = length(unique(d$dkey)), n_rows = nrow(d),
        n_after_present = sum(d$prev == 1),
        or_carry_over = o$or_per_unit, ci95_low = o$ci95_low, ci95_high = o$ci95_high,
        z = o$z, p = o$p, optimizer = optimizer, note = note,
        stringsAsFactors = FALSE
      )
    }
  }
  adjust_within_model(do.call(rbind, rows))
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

run_all <- function(msg, cl, er, pronoun) {
  list(
    h1_primary = run_primary(msg),
    cue_trends = run_cue_trends(cl, pronoun),
    emergence_model = run_emergence(er),
    persistence_model = run_persistence(cl, pronoun)
  )
}

main <- function(dir) {
  rd <- function(f) read.csv(file.path(dir, f), stringsAsFactors = FALSE, check.names = FALSE)
  pronoun_file <- file.path(dir, "pronoun_cue.txt")
  pronoun <- if (file.exists(pronoun_file)) trimws(readLines(pronoun_file, warn = FALSE)[1]) else ""
  out <- run_all(rd("messages.csv"), rd("cue_long.csv"), rd("emergence_risk.csv"), pronoun)
  for (name in names(out)) {
    write.csv(out[[name]], file.path(dir, paste0(name, ".csv")), row.names = FALSE)
  }
  cat("\nH1 (primary, one-sided, Holm across models):\n")
  print(out$h1_primary[, c("model", "or_per_turn", "ci95_low", "ci95_high",
                           "p_one_sided", "p_one_sided_holm", "h1_supported", "note")])
  cat("\nWrote h1_primary.csv, cue_trends.csv, emergence_model.csv, persistence_model.csv to", dir, "\n")
}

# ---------------------------------------------------------------------------
# Self-test on simulated data with known effects
# ---------------------------------------------------------------------------

selftest <- function() {
  set.seed(42)
  cues <- paste0("cue", 1:6)
  base <- seq(-2.5, -0.5, length.out = length(cues))

  # Each dialogue has a baseline u; warmer dialogues run longer (selection).
  sim <- function(model, beta, gamma, n_dial = 200) {
    u <- rnorm(n_dial)
    L <- pmin(pmax(round(7.5 + 2 * u), 5), 10)
    msg <- vector("list", sum(L))
    cl <- vector("list", sum(L))
    r <- 0
    for (i in seq_len(n_dial)) {
      prev <- rep(0L, length(cues))
      first <- rep(NA_integer_, length(cues))
      for (t in seq_len(L[i])) {
        p <- plogis(base + u[i] + beta * (t - 5.5) + gamma * prev)
        pres <- rbinom(length(cues), 1, p)
        if (t == 1) first <- pres
        r <- r + 1
        key <- paste(model, i, sep = "::")
        cl[[r]] <- data.frame(model = model, dkey = key, turn = t, length = L[i], cue = cues,
                              present = pres, prev = if (t == 1) NA_integer_ else prev,
                              first = first, stringsAsFactors = FALSE)
        msg[[r]] <- data.frame(model = model, dkey = key, turn = t, length = L[i],
                               k = sum(pres), n = length(cues), stringsAsFactors = FALSE)
        prev <- pres
      }
    }
    list(msg = do.call(rbind, msg), cl = do.call(rbind, cl))
  }

  risk_table <- function(cl) {
    key <- paste(cl$dkey, cl$cue, sep = "|")
    seen_turn <- ifelse(cl$present == 1, cl$turn, NA)
    fs <- tapply(seen_turn, key, function(x) if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE))
    cl$first_seen <- as.numeric(fs[key])
    r <- cl[cl$turn >= 2 & (is.na(cl$first_seen) | cl$turn <= cl$first_seen), ]
    r$event <- as.integer(!is.na(r$first_seen) & r$turn == r$first_seen)
    r[, c("model", "dkey", "cue", "turn", "length", "event")]
  }

  cat("Simulating: null (no change), up (log-odds +0.15 per turn), carry (carry-over, log-odds +1.2)\n")
  sims <- list(sim("null", 0, 0), sim("up", 0.15, 0), sim("carry", 0, 1.2))
  msg <- do.call(rbind, lapply(sims, function(s) s$msg))
  cl <- do.call(rbind, lapply(sims, function(s) s$cl))
  er <- risk_table(cl)
  out <- run_all(msg, cl, er, pronoun = "")

  h1 <- out$h1_primary
  em <- out$emergence_model
  pe <- out$persistence_model
  cat("\nH1:\n");          print(h1[, c("model", "or_per_turn", "ci95_low", "ci95_high", "p_one_sided_holm", "note")])
  cat("\nEmergence:\n");   print(em[, c("model", "or_per_turn", "ci95_low", "ci95_high", "note")])
  cat("\nPersistence (mean log OR over cues):\n")
  print(tapply(log(pe$or_carry_over), pe$model, mean, na.rm = TRUE))

  lor <- function(df, m, col) log(df[[col]][df$model == m])
  pe_mean <- function(m) mean(log(pe$or_carry_over[pe$model == m]), na.rm = TRUE)
  checks <- c(
    "H1, null: log OR per turn within +/-0.07 of 0" = abs(lor(h1, "null", "or_per_turn")) < 0.07,
    "H1, up: log OR per turn between 0.09 and 0.21 (true 0.15)" =
      lor(h1, "up", "or_per_turn") > 0.09 && lor(h1, "up", "or_per_turn") < 0.21,
    "Emergence, null: log OR per turn within +/-0.15 of 0" = abs(lor(em, "null", "or_per_turn")) < 0.15,
    "Emergence, up: OR per turn above 1" = lor(em, "up", "or_per_turn") > 0,
    "Persistence, null: mean log OR within +/-0.4 of 0" = abs(pe_mean("null")) < 0.4,
    "Persistence, carry: mean log OR above log(2) (true 1.2)" = pe_mean("carry") > log(2)
  )
  cat("\n")
  for (nm in names(checks)) cat(if (isTRUE(checks[[nm]])) "PASS " else "CHECK ", nm, "\n", sep = "")
  cat("\nThese are sanity checks with wide tolerances on one simulated sample; a single CHECK\n",
      "is worth a look, several mean something is wrong.\n", sep = "")
}

# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && args[1] == "--selftest") {
  selftest()
} else {
  main(if (length(args) >= 1) args[1] else "turn_dynamics_results")
}
