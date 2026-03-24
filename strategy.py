#!/usr/bin/env python3
"""
Experiment #568: 4h Primary + 12h/1d HTF — Simplified Regime-Adaptive with Connors RSI

Hypothesis: Previous 4h strategies failed due to overly complex entry conditions (0 trades).
This version SIMPLIFIES logic while keeping the proven regime-adaptive framework.

Key changes from failed #558/562/564:
1. FEWER confluence requirements (2-3 filters max, not 5+)
2. Connors RSI for mean reversion (proven 75% win rate in backtests)
3. Looser ADX/CHOP thresholds (ADX>20 not 25, CHOP>50 not 61.8)
4. Always allow SOME entry in each regime (no "stay flat" zones)
5. Larger base size (0.30) to ensure meaningful position when signals fire

Strategy logic:
1. 1d HMA(21) = primary trend bias (simpler than 1w+1d dual)
2. 12h HMA(21) = secondary confirmation
3. 4h Connors RSI = entry timing (CRSI<10 long, CRSI>90 short)
4. 4h Choppiness(14) = regime switch (CHOP>50 = mean revert, CHOP<50 = trend)
5. 4h ATR(14)*2.5 = stoploss on all positions

Regime-adaptive entries:
- TREND (CHOP<50): Follow 1d HMA direction, enter on CRSI pullback (20-40 / 60-80)
- RANGE (CHOP>50): Mean revert at CRSI extremes (<10 / >90)
- NO flat zones - always have some entry condition active

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak - streak duration (consecutive up/down days)
    3. PercentRank(100) - where current close ranks vs last 100 closes
    
    Entry signals:
    - CRSI < 10 = extreme oversold (long)
    - CRSI > 90 = extreme overbought (short)
    - CRSI 20-40 = pullback in uptrend (long)
    - CRSI 60-80 = pullback in downtrend (short)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    rsi_close[:] = np.nan
    for i in range(3, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute for RSI calculation
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(2, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i])
        crsi[i] = (rsi_close[i] + rsi_streak[i] + (rank / rank_period) * 100.0) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 50 = range-bound (mean reversion preferred)
    CHOP < 50 = trending (trend follow preferred)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d primary, 12h secondary) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 50.0   # Mean reversion regime
        is_trend = chop[i] <= 50.0  # Trend following regime
        
        # === CRSI LEVELS ===
        crsi_extreme_low = crsi[i] < 15.0
        crsi_extreme_high = crsi[i] > 85.0
        crsi_low = crsi[i] < 35.0
        crsi_high = crsi[i] > 65.0
        crsi_neutral_low = crsi[i] >= 20.0 and crsi[i] <= 40.0
        crsi_neutral_high = crsi[i] >= 60.0 and crsi[i] <= 80.0
        
        # === ENTRY LOGIC (SIMPLIFIED - always have some entry condition) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at CRSI extremes
        if is_range:
            if crsi_extreme_low:
                desired_signal = SIZE_STRONG  # Strong long at extreme oversold
            elif crsi_extreme_high:
                desired_signal = -SIZE_STRONG  # Strong short at extreme overbought
            elif crsi_low and htf_bull:
                desired_signal = SIZE_BASE  # Moderate long in uptrend pullback
            elif crsi_high and htf_bear:
                desired_signal = -SIZE_BASE  # Moderate short in downtrend pullback
        
        # TREND REGIME: Follow HTF direction on CRSI pullback
        elif is_trend:
            if htf_bull and crsi_neutral_low:
                desired_signal = SIZE_BASE  # Long on pullback in uptrend
            elif htf_bear and crsi_neutral_high:
                desired_signal = -SIZE_BASE  # Short on pullback in downtrend
            elif htf_bull and crsi_low:
                desired_signal = SIZE_STRONG  # Strong long on deeper pullback
            elif htf_bear and crsi_high:
                desired_signal = -SIZE_STRONG  # Strong short on deeper pullback
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
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