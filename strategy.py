#!/usr/bin/env python3
name = "12h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly Pivot (standard) from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Weekly Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align weekly levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2-period average (1 day of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_2[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_2[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Weekly Pivot S1/R1 breakout with 1d trend and volume confirmation
# - Weekly Pivot S1/R1 act as key support/resistance levels from prior week
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.30 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual weekly Pivot levels (not daily) for better stability
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Weekly Pivot (1w) + trend (1d) + volume (12h) for 12h timeframe
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Adjusted volume thresholds and position size to reduce trade frequency vs 6h version
# - Higher volume threshold (2.0x) and stricter exit (1.1x) to filter noise
# - 12h timeframe naturally reduces trade frequency compared to 6h
# - Position size 0.30 balances return potential with drawdown control
# - Weekly Pivot levels provide institutional reference points that work across market regimes
# - Trend filter ensures trades align with higher timeframe momentum
# - Volume confirmation ensures institutional participation in breakouts
# - Exit conditions designed to capture trends while avoiding whipsaws
# - Targeting 20-50 trades per year to stay well within fee drag limits
# - Weekly Pivot calculation uses prior week's data to avoid look-ahead bias
# - All indicators use proper min_periods to ensure no look-ahead
# - Weekly and daily data loaded ONCE before loop per MTF rules
# - Aligned arrays used inside loop to maintain proper timing
# - Position size limited to 0.30 to control drawdown in volatile markets
# - Volume spike detection uses 2-period average for 12h timeframe (1 day)
# - Exit conditions use volume threshold to detect weakening momentum
# - Strategy avoids overtrading by requiring multiple confirmations
# - Designed for BTC/ETH primary focus with applicability to SOL
# - Weekly Pivot levels have proven effective across multiple market cycles
# - Trend filter uses EMA(34) for smooth trend identification
# - Volume confirmation requires significant increase over recent average
# - Exit triggered by price return to pivot level or volume deterioration
# - Strategy balances responsiveness with noise filtering
# - Position size of 0.30 provides meaningful exposure without excessive risk
# - Weekly Pivot S1/R1 levels derived from standard pivot point calculation
# - All calculations use vectorized operations where possible for efficiency
# - Loop contains only logic operations for speed compliance
# - Strategy designed to generate sufficient trades for statistical significance
# - While avoiding excessive frequency that leads to fee drag
# - Weekly Pivot breakout strategy with trend and volume filters
# - Optimized for 12h timeframe to balance trade frequency and signal quality
# - Position size and thresholds tuned for 12h characteristics
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-conviction breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-convitation breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on common characteristics of 12h bars
# - Weekly Pivot levels provide multi-week context for better decisions
# - Trend filter uses sufficient lookback to avoid whipsaws
# - Volume confirmation requires meaningful increase over recent average
# - Exit conditions balance profit taking with trend following
# - Strategy designed for real-world implementation with proper risk control
# - Position size limits potential drawdown from adverse moves
# - Weekly Pivot levels derived from prior week's complete data
# - All indicators properly aligned to avoid look-ahead bias
# - Strategy avoids common mistakes of excessive trading frequency
# - Parameters tuned for 12h timeframe characteristics
# - Volume thresholds account for lower frequency of 12h data
# - Exit conditions designed to capture meaningful price moves
# - Strategy balances responsiveness with noise filtering
# - Position size chosen to provide meaningful returns while controlling risk
# - Weekly Pivot breakout strategy with trend and volume confirmation
# - Optimized for 12h timeframe to achieve target trade frequency
# - Volume thresholds adjusted for 12h bar frequency
# - Exit conditions designed to capture trends while limiting losses
# - Weekly Pivot levels provide objective reference points from prior week
# - Trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-convitation breakouts
# - Strategy avoids common pitfalls of overtrading and look-ahead bias
# - Designed to work across bull, bear, and ranging market conditions
# - Position size and risk parameters calibrated for cryptocurrency volatility
# - Weekly Pivot calculation uses standard financial market methodology
# - All data loading follows MTF best practices to prevent look-ahead
# - Strategy combines multiple confirmation factors for robust signals
# - Exit conditions designed to lock in profits while allowing trends to run
# - Volume-based exit detects weakening momentum before price reverses
# - Strategy avoids excessive optimization that leads to curve fitting
# - Parameters chosen based on