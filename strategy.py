#!/usr/bin/env python3
"""
Experiment #1475: 6h Primary + 1d/1w HTF — Adaptive Trend Following with Pullback Entries

Hypothesis: 6h timeframe captures multi-day crypto swings better than 4h (too noisy) or 12h (too slow).
This strategy uses KAMA (Kaufman Adaptive Moving Average) which adapts to volatility - 
faster in trends, slower in chop. Combined with 1w/1d HMA trend filters for major direction.

Key components:
1. 1w HMA(21) = secular trend bias (NEVER trade counter to this)
2. 1d HMA(21) = intermediate trend confirmation
3. 6h KAMA(14) = adaptive entry trigger (pullback to KAMA in trend direction)
4. 6h ROC(10) = momentum confirmation (avoid exhausted moves)
5. 6h ATR(14) = volatility filter + trailing stoploss (2.5x ATR)
6. Volatility-based position sizing (reduce size in high vol)

Why this should work on 6h:
- KAMA adapts to crypto's variable volatility better than fixed EMA/HMA
- 1w filter prevents major counter-trend disasters (2022 crash, 2021 blowoff)
- Pullback entries = better risk/reward than breakout chasing
- 6h TF = natural 30-50 trades/year (fee-efficient)
- LOOSE entry thresholds guarantee trades (KAMA within 2%, ROC just positive/negative)

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + 1d_HMA bullish + price pulls back to KAMA + ROC > 0
- SHORT: 1w_HMA bearish + 1d_HMA bearish + price rallies to KAMA + ROC < 0

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete (volatility scaled)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_pullback_hma_1d1w_v1"
timeframe = "6h"
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

def calculate_kama(close, period=14, fast_sc=2.0/(14+1), slow_sc=2.0/(30+1)):
    """
    Kaufman Adaptive Moving Average
    Adapts to market noise - fast in trends, slow in chop
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # First KAMA value = close
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(close[i]):
            continue
        
        # Calculate change over period
        change = abs(close[i] - close[i - period])
        
        # Calculate sum of individual changes (noise)
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        # Efficiency Ratio (0 = noise, 1 = pure trend)
        if noise > 1e-10:
            er = change / noise
        else:
            er = 1.0
        
        # Smoothed constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    kama_14 = calculate_kama(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(roc_10[i]):
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
        
        # === TREND DIRECTION (1w + 1d HMA bias) ===
        # 1w = secular trend, 1d = intermediate trend
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA PULLBACK DETECTION ===
        # Price within 2% of KAMA = pullback zone
        kama_distance_pct = (close[i] - kama_14[i]) / kama_14[i] * 100 if kama_14[i] > 1e-10 else 0
        near_kama_long = kama_distance_pct >= -2.0 and kama_distance_pct <= 1.0
        near_kama_short = kama_distance_pct >= -1.0 and kama_distance_pct <= 2.0
        
        # === MOMENTUM (ROC) ===
        roc = roc_10[i]
        momentum_positive = roc > 0.5  # slight positive momentum
        momentum_negative = roc < -0.5  # slight negative momentum
        
        # === VOLATILITY FILTER ===
        # Reduce position size in extreme volatility
        vol_ratio = atr_14[i] / close[i] * 100  # ATR as % of price
        if vol_ratio > 8.0:  # very high vol
            size_multiplier = 0.7
        elif vol_ratio > 5.0:  # high vol
            size_multiplier = 0.85
        else:
            size_multiplier = 1.0
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + pullback to KAMA + positive momentum
        if price_above_1w and price_above_1d and near_kama_long and momentum_positive:
            desired_signal = SIZE_STRONG * size_multiplier
        
        # SHORT: 1w bearish + 1d bearish + rally to KAMA + negative momentum
        elif price_below_1w and price_below_1d and near_kama_short and momentum_negative:
            desired_signal = -SIZE_STRONG * size_multiplier
        
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
        if desired_signal >= SIZE_STRONG * 0.8:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.8:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.8:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.8:
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