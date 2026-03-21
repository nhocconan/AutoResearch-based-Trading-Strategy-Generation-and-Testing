#!/usr/bin/env python3
"""
EXPERIMENT #030 - Daily EMA Crossover + Weekly HMA Trend + RSI Filter (1d primary)
=====================================================================================
Hypothesis: Daily timeframe captures major crypto trends while avoiding noise of lower TFs.
Using Weekly HMA(21) as trend filter ensures we trade with the dominant multi-week direction.
EMA(12/26) crossover provides clear entry signals on daily bars. RSI(14) filter avoids
entering at extremes (RSI > 70 long, RSI < 30 short = overbought/oversold reversal risk).
Bollinger Band Width percentile detects low-volatility regimes where breakouts are more reliable.

Key features:
- Primary TF: 1d (daily bars = ~1460 bars over 4 years, sufficient for 10+ trades)
- HTF filter: 1w HMA(21) for major trend direction
- Entry: EMA(12) crossing EMA(26) with momentum confirmation
- Filter: RSI(14) not at extremes (30-70 range for entries)
- Regime: BB Width > 40th percentile (avoid ultra-low vol chop)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should work on 1d:
- Daily bars have enough signal per bar (vs 15m noise)
- Weekly trend filter aligns with crypto's multi-week cycles
- EMA crossover on daily generates 20-50 signals/year = 80-200 over 4 years
- Conservative sizing (0.25-0.30) survives 2022 bear market
- BB regime filter avoids dead zones where crossovers whipsaw
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_cross_1whma_rsi_1d_v1"
timeframe = "1d"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma
    return upper, lower, bb_width


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_percentile_rank(arr, window=100):
    """Calculate rolling percentile rank"""
    n = len(arr)
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(window, n):
        window_vals = arr[i-window:i]
        pr[i] = np.sum(window_vals < arr[i]) / window
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend filter
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate daily indicators
    ema_fast = calculate_ema(close, 12)
    ema_slow = calculate_ema(close, 26)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_width_pr = calculate_percentile_rank(bb_width, window=100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size
    MIN_SIZE = 0.22   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(ema_fast[i]) or
            np.isnan(ema_slow[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(bb_width_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1w HMA trend filter
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        hma_trend = 1 if price_above_1w_hma else -1
        
        # EMA crossover signal
        ema_cross_bullish = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_bearish = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        ema_aligned_long = ema_fast[i] > ema_slow[i]
        ema_aligned_short = ema_fast[i] < ema_slow[i]
        
        # RSI filter (avoid extremes)
        rsi_ok_long = 35 < rsi[i] < 70  # Not overbought for long
        rsi_ok_short = 30 < rsi[i] < 65  # Not oversold for short
        
        # Bollinger Band Width regime (avoid ultra-low vol)
        bb_regime_ok = bb_width_pr[i] > 0.35  # Above 35th percentile
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: EMA cross bullish + 1w HMA bullish + RSI ok + BB regime ok
        if (ema_cross_bullish and hma_trend == 1 and rsi_ok_long and bb_regime_ok):
            target_signal = position_size
        
        # Short entry: EMA cross bearish + 1w HMA bearish + RSI ok + BB regime ok
        elif (ema_cross_bearish and hma_trend == -1 and rsi_ok_short and bb_regime_ok):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if EMA alignment reverses OR 1w HMA alignment breaks
                ema_reversal_long = ema_fast[i] < ema_slow[i]
                ema_reversal_short = ema_fast[i] > ema_slow[i]
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if ema_reversal_long or ema_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals