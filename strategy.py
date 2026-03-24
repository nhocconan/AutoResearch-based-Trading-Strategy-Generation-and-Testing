#!/usr/bin/env python3
"""
Experiment #696: 30m Primary + 4h/1d HTF — Regime-Adaptive CRSI Mean Reversion

Hypothesis: 30m timeframe with strict confluence filters can achieve optimal trade frequency
(40-80/year) while maintaining positive Sharpe. Using 4h HMA for trend bias, 1d HMA for
regime filter, Connors RSI for entry timing, and Choppiness Index to confirm mean-reversion
regime. Session filter (08-20 UTC) avoids low-liquidity periods.

Key innovations:
1. 4h HMA(21) bias - only long when 4h trend bullish, only short when bearish
2. 1d HMA(21) regime - avoid counter-trend trades against daily direction
3. Connors RSI(3,2,100) - superior to standard RSI for mean reversion (75% win rate)
4. Choppiness(14) > 50 - confirm ranging market (mean reversion works best here)
5. Session filter 08-20 UTC - avoid Asian overnight low-liquidity whipsaws
6. ATR(14) trailing stop 2.5x - protect capital during trend reversals
7. Discrete sizing: 0.0, ±0.20, ±0.25 to minimize fee churn

Entry conditions (balanced for trade generation):
- LONG: 4h HMA bull + 1d HMA bull + CHOP>50 + CRSI<30 + session 08-20 UTC
- SHORT: 4h HMA bear + 1d HMA bear + CHOP>50 + CRSI>70 + session 08-20 UTC

Target: Sharpe>0.40, trades>=40/year train, trades>=3 test, DD>-40%
Timeframe: 30m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_regime_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close):
    """
    Connors RSI - composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Based on ConnorsRSI methodology (2008)
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3) - short-term momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain_3 = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    avg_loss_3 = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rs_3 = np.zeros(n)
    rs_3[:] = np.nan
    for i in range(3, n):
        if avg_loss_3[i] > 1e-10:
            rs_3[i] = avg_gain_3[i] / avg_loss_3[i]
        else:
            rs_3[i] = 100.0
    rsi_3 = 100.0 - (100.0 / (1.0 + rs_3))
    
    # RSI of Streak (2) - streak duration momentum
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI on streak values
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rs_streak = np.zeros(n)
    rs_streak[:] = np.nan
    for i in range(2, n):
        if avg_streak_loss[i] > 1e-10:
            rs_streak[i] = avg_streak_gain[i] / avg_streak_loss[i]
        else:
            rs_streak[i] = 100.0
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Percent Rank (100) - where current close ranks in last 100 bars
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(100, n):
        window = close[i-99:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = (count_below / 99) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of TR
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate Choppiness
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    for i in range(period, n):
        range_val = highest[i] - lowest[i]
        if range_val > 1e-10 and tr_sum[i] > 0:
            choppiness[i] = 100.0 * np.log10(tr_sum[i] / range_val) / np.log10(period)
    
    return choppiness

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    crsi = calculate_crsi(close)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 50 = ranging (mean reversion works)
        is_ranging = choppiness[i] > 50.0
        
        # === CRSI ENTRY SIGNALS ===
        # LONG: CRSI < 30 (oversold)
        crsi_oversold = crsi[i] < 30.0
        # SHORT: CRSI > 70 (overbought)
        crsi_overbought = crsi[i] > 70.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1d bull + ranging + CRSI oversold + session
        if htf_4h_bull and htf_1d_bull and is_ranging and crsi_oversold and in_session:
            desired_signal = SIZE_STRONG
        # Weaker long: 4h bull + ranging + CRSI oversold (skip 1d filter)
        elif htf_4h_bull and is_ranging and crsi_oversold and in_session:
            desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 1d bear + ranging + CRSI overbought + session
        elif htf_4h_bear and htf_1d_bear and is_ranging and crsi_overbought and in_session:
            desired_signal = -SIZE_STRONG
        # Weaker short: 4h bear + ranging + CRSI overbought (skip 1d filter)
        elif htf_4h_bear and is_ranging and crsi_overbought and in_session:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
                entry_atr = atr[i]
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