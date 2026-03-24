#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: After #001 failed (Donchian breakout), switch to Connors RSI mean reversion
which proved Sharpe +0.923 on ETH in research. Key insight: crypto spends 60-70% of
time in range/chop, so mean reversion should outperform trend following in bear/range markets.

Strategy Components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Captures short-term oversold/overbought better than standard RSI(14)
2. Choppiness Index (CHOP): Regime filter
   - CHOP > 61.8 = range regime → use mean reversion (CRSI extremes)
   - CHOP < 38.2 = trend regime → use HTF direction only
3. 1d HMA: Primary trend bias (price vs HMA alignment)
4. 1w HMA: Macro trend confirmation (avoid counter-macro trades)
5. ATR(14) 2.5x trailing stop: Proven risk management

Entry Logic:
- Range Regime (CHOP>61.8):
  - Long: CRSI < 20 + price > 1d HMA
  - Short: CRSI > 80 + price < 1d HMA
- Trend Regime (CHOP<38.2):
  - Long: price > 1d HMA + price > 1w HMA
  - Short: price < 1d HMA + price < 1w HMA

Why this differs from #001:
- Mean reversion vs breakout (opposite philosophy)
- Connors RSI vs standard RSI (more sensitive to short-term extremes)
- Choppiness regime switch (adapts to market conditions)
- Dual HTF (1d + 1w) for stronger trend confirmation

Risk: 2.5x ATR trailing stop, discrete sizing 0.25
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Based on Connors & Alvarez (2008) - captures short-term mean reversion signals
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rsi3 = np.full(n, np.nan)
    for i in range(3, n):
        if avg_loss[i] < 1e-10:
            rsi3[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
        
        # RSI of streak (simplified: map streak to 0-100)
        if i >= 2:
            if streak[i] > 0:
                streak_rsi[i] = min(100, 50 + streak[i] * 10)
            elif streak[i] < 0:
                streak_rsi[i] = max(0, 50 + streak[i] * 10)
            else:
                streak_rsi[i] = 50
    
    # Percent Rank (100) - where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(100, n):
        window = close[i-100:i+1]
        current = close[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    for i in range(100, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies ranging vs trending markets
    Formula: 100 * (ATR(1,sum of TR) / (Highest High - Lowest Low)) / log10(period)
    CHOP > 61.8 = range, CHOP < 38.2 = trend
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            choppiness[i] = 100.0 * (tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    
    # Calculate and align HTF HMAs for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_crsi(close)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI (100) + CHOP (14) + warm-up
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range_regime = choppiness[i] > 61.8
        is_trend_regime = choppiness[i] < 38.2
        
        # === HTF TREND BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_range_regime:
            # Mean Reversion Strategy (CRSI extremes)
            # Long: CRSI < 20 (oversold) + price above 1d HMA
            if crsi[i] < 20.0 and hma_1d_bull:
                desired_signal = SIZE
            
            # Short: CRSI > 80 (overbought) + price below 1d HMA
            elif crsi[i] > 80.0 and hma_1d_bear:
                desired_signal = -SIZE
        
        elif is_trend_regime:
            # Trend Following Strategy (HTF alignment)
            # Long: price > 1d HMA + price > 1w HMA (both bullish)
            if hma_1d_bull and hma_1w_bull:
                desired_signal = SIZE
            
            # Short: price < 1d HMA + price < 1w HMA (both bearish)
            elif hma_1d_bear and hma_1w_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals