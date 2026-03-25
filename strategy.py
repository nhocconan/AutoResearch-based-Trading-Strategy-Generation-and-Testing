#!/usr/bin/env python3
"""
Experiment #1248: 4h Primary + 12h/1d HTF — Dual Regime Strategy

Hypothesis: Single-regime strategies fail because markets alternate between trending
and ranging. This strategy uses Choppiness Index to detect regime and switches logic:

REGIME DETECTION (Choppiness Index 14):
- CHOP > 55 = Ranging market → Mean reversion entries (CRSI extremes)
- CHOP < 45 = Trending market → Trend pullback entries (CRSI moderate)
- 45-55 = Transition → Reduce size or stay flat

ENTRY LOGIC:
- Ranging: CRSI < 20 long, CRSI > 80 short (mean reversion at bounds)
- Trending: CRSI < 45 long in uptrend, CRSI > 55 short in downtrend

HTF CONFIRMATION:
- 1d HMA(21) for primary trend bias
- 12h HMA(21) for intermediate confirmation
- Align both HTFs properly using mtf_data helper

RISK:
- ATR(14) 2.5x trailing stop
- Discrete sizing: 0.0, ±0.25, ±0.30
- Position tracking for stoploss management

Why this should beat 6h_kama (Sharpe=0.447):
- Regime adaptation reduces whipsaw losses in 2022 crash
- CRSI more responsive than standard RSI for entry timing
- 4h TF balances trade frequency (30-60/year) vs fee drag
- Dual HTF confirmation reduces false signals

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_hma_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index - measures market consolidation vs trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Ranging | CHOP < 38.2 = Trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        tr_sum = np.nansum(tr[i - period + 1:i + 1])
        hh = np.nanmax(high[i - period + 1:i + 1])
        ll = np.nanmin(low[i - period + 1:i + 1])
        range_hl = hh - ll
        if range_hl > 1e-10 and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_rsi_streak(close, period=2):
    """
    RSI of consecutive up/down streaks (Connors RSI component)
    Streak = number of consecutive days price moved in same direction
    """
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            if i > 1 and close[i-1] > close[i-2]:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if i > 1 and close[i-1] < close[i-2]:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_abs = np.abs(streak)
    gain = np.where(streak > 0, streak_abs, 0.0)
    loss = np.where(streak < 0, streak_abs, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs))
    rsi_streak[:period+1] = np.nan
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """
    Percentile Rank - where current price ranks in last N periods
    (Connors RSI component)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = 100.0 * np.sum(valid < close[i]) / len(valid)
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Calculate 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_3 = calculate_rsi(close, period=3)
    rsi_streak = calculate_rsi_streak(close, period=2)
    pr_100 = calculate_percent_rank(close, period=100)
    chop_14 = calculate_chop(high, low, close, period=14)
    
    # Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr_100[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + pr_100[i]) / 3.0
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_ranging = chop > 50.0  # More lenient threshold
        is_trending = chop < 50.0
        
        # === TREND DIRECTION (Daily + 12h HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        hma_12h_valid = not np.isnan(hma_12h_aligned[i])
        price_above_12h = hma_12h_valid and close[i] > hma_12h_aligned[i]
        price_below_12h = hma_12h_valid and close[i] < hma_12h_aligned[i]
        
        # Strong trend = both 12h and 1d aligned
        strong_uptrend = price_above_1d and price_above_12h
        strong_downtrend = price_below_1d and price_below_12h
        
        # === ENTRY LOGIC (Dual Regime) ===
        desired_signal = 0.0
        crsi_val = crsi[i]
        
        if is_ranging:
            # MEAN REVERSION: Enter at extremes
            # Long: CRSI < 25 (oversold in range)
            if crsi_val < 25.0:
                if price_above_1d:
                    desired_signal = SIZE_STRONG  # Range low + uptrend bias
                else:
                    desired_signal = SIZE_BASE  # Pure mean reversion
            
            # Short: CRSI > 75 (overbought in range)
            elif crsi_val > 75.0:
                if price_below_1d:
                    desired_signal = -SIZE_STRONG  # Range high + downtrend bias
                else:
                    desired_signal = -SIZE_BASE  # Pure mean reversion
        
        else:
            # TREND FOLLOWING: Enter on pullbacks
            # Long: Uptrend + CRSI pullback (30-55)
            if strong_uptrend and 30.0 <= crsi_val <= 55.0:
                desired_signal = SIZE_STRONG
            elif price_above_1d and 35.0 <= crsi_val <= 50.0:
                desired_signal = SIZE_BASE
            
            # Short: Downtrend + CRSI pullback (45-70)
            elif strong_downtrend and 45.0 <= crsi_val <= 70.0:
                desired_signal = -SIZE_STRONG
            elif price_below_1d and 50.0 <= crsi_val <= 65.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals