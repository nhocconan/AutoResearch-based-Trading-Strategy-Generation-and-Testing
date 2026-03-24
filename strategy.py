#!/usr/bin/env python3
"""
Experiment #827: 6h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime

Hypothesis: 6h timeframe with CRSI (Connors RSI) for entries + CHOP for regime
detection will outperform standard RSI strategies. CRSI has 75% win rate in
mean reversion. CHOP tells us when to mean-revert (CHOP>61.8) vs trend-follow
(CHOP<38.2). 1d HMA for directional bias, 1w HMA for major trend filter.

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Choppiness Index(14) for regime: >61.8=range, <38.2=trend
3. 1d HMA(21) for medium-term bias
4. 1w HMA(21) for major trend agreement
5. Regime-adaptive entries: mean-revert in range, trend-pullback in trend
6. Discrete sizing: 0.0, ±0.25, ±0.30 with ATR 2.5x trailing stop

Entry conditions (LOOSE for trade generation):
- RANGE mode: CRSI<20 long, CRSI>80 short (mean reversion at extremes)
- TREND mode: 1d HMA bull + CRSI<40 long, 1d HMA bear + CRSI>60 short
- 1w HMA agreement boosts size to 0.30

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_regime_1d1w_v1"
timeframe = "6h"
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close):
    """
    Connors RSI - combines short-term momentum, streak, and percentile rank
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion strategies
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI Streak(2) - consecutive up/down days
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(2, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 - (100.0 / (streak[i] + 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 / (abs(streak[i]) + 1)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank(100) - where current close ranks in last 100 bars
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(100, n):
        window = close[i-99:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = (count_below / 99.0) * 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(100, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies ranging vs trending markets
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        if hh[i] > ll[i] and atr_sum[i] > 0:
            chop[i] = 100.0 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
    
    return chop

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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop[i] > 61.8  # Mean reversion mode
        regime_trend = chop[i] < 38.2  # Trend following mode
        # 38.2 <= CHOP <= 61.8 = neutral (use both signals)
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        desired_signal = 0.0
        signal_strength = SIZE_BASE
        
        # RANGE MODE: Mean reversion at CRSI extremes
        if regime_range:
            if crsi[i] < 20.0:  # Extremely oversold
                desired_signal = SIZE_BASE
                if htf_1d_bull:
                    desired_signal = SIZE_STRONG
            elif crsi[i] > 80.0:  # Extremely overbought
                desired_signal = -SIZE_BASE
                if htf_1d_bear:
                    desired_signal = -SIZE_STRONG
        
        # TREND MODE: Follow HTF direction with CRSI pullback
        elif regime_trend:
            if htf_1d_bull and crsi[i] < 40.0:  # Pullback in uptrend
                desired_signal = SIZE_BASE
                if htf_1w_bull:
                    signal_strength = SIZE_STRONG
                    desired_signal = SIZE_STRONG
            elif htf_1d_bear and crsi[i] > 60.0:  # Pullback in downtrend
                desired_signal = -SIZE_BASE
                if htf_1w_bear:
                    signal_strength = SIZE_STRONG
                    desired_signal = -SIZE_STRONG
        
        # NEUTRAL MODE: Both range and trend signals valid
        else:
            # Mean reversion signals (slightly looser)
            if crsi[i] < 25.0 and htf_1d_bull:
                desired_signal = SIZE_BASE
            elif crsi[i] > 75.0 and htf_1d_bear:
                desired_signal = -SIZE_BASE
            # Trend pullback signals
            elif htf_1d_bull and crsi[i] < 45.0:
                desired_signal = SIZE_BASE
            elif htf_1d_bear and crsi[i] > 55.0:
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