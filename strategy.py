#!/usr/bin/env python3
"""
Experiment #047: 6h Primary + 1d HTF — Volatility Spike Mean Reversion + Z-Score

Hypothesis: After 46 failed experiments, the pattern is clear:
- Trend following fails on 6h (too many whipsaws in bear/range markets)
- Pure mean reversion needs volatility confirmation to avoid catching falling knives
- SOLUTION: Volatility spike (ATR ratio) + Bollinger extreme + Z-score confluence
- This captures "vol crush" after panic selling (2022 crash, 2025 bear)
- 1d HMA provides major trend bias but we allow counter-trend mean reversion at extremes
- Z-score(20) of price ensures we're at statistical extremes
- This is DIFFERENT from all previous 6h attempts (no cRSI, no Fisher, no Choppiness)

Key design choices:
- Timeframe: 6h (30-60 trades/year target, middle ground between 4h and 12h)
- HTF: 1d HMA(50) for major trend bias
- Entry: ATR(7)/ATR(30) > 1.8 (vol spike) + price outside BB(20, 2.2) + Z-score > 2.0
- Exit: ATR ratio < 1.3 (vol normalized) or price crosses BB mid
- Position size: 0.30 (30% of capital, conservative for mean reversion)
- Stoploss: 3.0x ATR trailing (wider for mean reversion whipsaws)
- LOOSE enough filters to ensure >=30 trades on train, >=3 on test

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_bb_zscore_1d_v1"
timeframe = "6h"
leverage = 1.0

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
    """Average True Range for volatility measurement"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_zscore(close, period=20):
    """Z-score of price relative to rolling mean"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.2)
    zscore = calculate_zscore(close, period=20)
    
    # ATR ratio for volatility spike detection
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]):
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for mean reversion)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_mid[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(zscore[i]) or np.isnan(atr_ratio[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30) = panic
        
        # === BOLLINGER EXTREME ===
        bb_extreme_low = close[i] < bb_lower[i]
        bb_extreme_high = close[i] > bb_upper[i]
        
        # === Z-SCORE EXTREME ===
        zscore_oversold = zscore[i] < -2.0
        zscore_overbought = zscore[i] > 2.0
        
        # === DESIRED SIGNAL (Volatility Spike Mean Reversion) ===
        desired_signal = 0.0
        
        # LONG: vol spike + BB lower + Z-score oversold
        # Allow counter-trend if extreme enough (mean reversion edge)
        if vol_spike and bb_extreme_low and zscore_oversold:
            desired_signal = SIZE
        elif vol_spike and bb_extreme_low and htf_bull:
            # Weaker signal: vol spike + BB lower + HTF bull (no Z-score req)
            desired_signal = SIZE * 0.7
        elif zscore[i] < -2.5 and htf_bull:
            # Fallback: extreme Z-score + HTF bull
            desired_signal = SIZE * 0.7
        
        # SHORT: vol spike + BB upper + Z-score overbought
        if vol_spike and bb_extreme_high and zscore_overbought:
            desired_signal = -SIZE
        elif vol_spike and bb_extreme_high and htf_bear:
            # Weaker signal: vol spike + BB upper + HTF bear
            desired_signal = -SIZE * 0.7
        elif zscore[i] > 2.5 and htf_bear:
            # Fallback: extreme Z-score + HTF bear
            desired_signal = -SIZE * 0.7
        
        # === EXIT CONDITIONS (Vol normalized or price reverts) ===
        if in_position:
            vol_normalized = atr_ratio[i] < 1.3
            price_reverted_long = close[i] > bb_mid[i]
            price_reverted_short = close[i] < bb_mid[i]
            
            if position_side > 0 and (vol_normalized or price_reverted_long):
                desired_signal = 0.0
            elif position_side < 0 and (vol_normalized or price_reverted_short):
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
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