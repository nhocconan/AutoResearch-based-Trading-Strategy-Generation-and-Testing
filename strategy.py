#!/usr/bin/env python3
"""
Experiment #743: 6h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: 6h timeframe with Connors RSI (CRSI) for mean reversion entries,
filtered by 1w HMA trend bias and 1d Choppiness Index regime detection.
CRSI has proven 75% win rate in range markets. 1w HMA provides ultra-slow
trend bias to avoid counter-trend mean reversion. 1d CHOP filters for
range vs trending regimes.

Key innovations:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. 1w HMA(21) for ultra-HTF trend bias (flips rarely = fewer bad trades)
3. 1d Choppiness Index (CHOP) for regime: >50 = range (mean revert OK)
4. 6h HMA(16/48) for local trend confirmation
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure trades):
- LONG: 1w HMA bull + 1d CHOP>45 + CRSI<20 + 6h HMA not strongly bear
- SHORT: 1w HMA bear + 1d CHOP>45 + CRSI>80 + 6h HMA not strongly bull

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_regime_1w1d_v1"
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
    Connors RSI (CRSI) - combines 3 components for mean reversion signal
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): very short-term momentum
    RSI_Streak(2): RSI of consecutive up/down day streaks
    PercentRank(100): where current close ranks in last 100 bars
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # Component 2: RSI of Streak (2)
    # Streak = consecutive up/down days count
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak values (period=2)
    streak_rsi = calculate_rsi(np.abs(streak), period=2)
    # Adjust sign based on streak direction
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(100, n):
        window = close[i-99:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100.0 * rank / 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
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
    
    # Highest High - Lowest Low over period
    hh_ll = pd.Series(high).rolling(window=period, min_periods=period).max().values - \
            pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # CHOP formula
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        if hh_ll[i] > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / hh_ll[i]) / np.log10(period)
    
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
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    chop_1d_raw = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    crsi = calculate_crsi(close)
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME FILTER (1d CHOP) ===
        # CHOP > 45 = range market (mean reversion works)
        # CHOP < 45 = trending (avoid mean reversion)
        regime_range = chop_1d_aligned[i] > 45.0
        
        # === 6h HMA TREND ===
        hma_6h_bull = hma_16[i] > hma_48[i]
        hma_6h_bear = hma_16[i] < hma_48[i]
        hma_spread = (hma_16[i] - hma_48[i]) / hma_48[i] if hma_48[i] > 1e-10 else 0
        
        # === CRSI CONDITIONS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 20.0  # Extreme oversold
        crsi_overbought = crsi[i] > 80.0  # Extreme overbought
        crsi_moderate_oversold = crsi[i] < 30.0
        crsi_moderate_overbought = crsi[i] > 70.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 1w bull + range regime + CRSI oversold + 6h not strongly bear
        if htf_1w_bull and regime_range:
            if crsi_oversold:
                # Strong signal: extreme CRSI
                if not hma_6h_bear or hma_spread > -0.02:
                    desired_signal = SIZE_STRONG
            elif crsi_moderate_oversold:
                # Moderate signal: needs 6h HMA confirm
                if hma_6h_bull or hma_spread > -0.01:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1w bear + range regime + CRSI overbought + 6h not strongly bull
        elif htf_1w_bear and regime_range:
            if crsi_overbought:
                # Strong signal: extreme CRSI
                if not hma_6h_bull or hma_spread < 0.02:
                    desired_signal = -SIZE_STRONG
            elif crsi_moderate_overbought:
                # Moderate signal: needs 6h HMA confirm
                if hma_6h_bear or hma_spread < 0.01:
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