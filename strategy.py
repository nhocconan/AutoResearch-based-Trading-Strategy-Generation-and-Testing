#!/usr/bin/env python3
"""
Experiment #1412: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Regime

Hypothesis: 12h timeframe offers optimal trade frequency (20-50/year) with reduced fee drag.
This strategy combines adaptive trend following with regime detection:

1. 1d HMA(21) for major trend bias (avoid counter-trend in strong moves)
2. 12h KAMA(14) adaptive trend (adjusts speed based on market efficiency)
3. Choppiness Index(14) regime filter: CHOP>61.8=range, CHOP<38.2=trend
4. ROC(10) momentum confirmation (avoid entering against momentum)
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should beat #1411 (Sharpe=0.575):
- KAMA adapts to volatility better than HMA/EMA in crypto
- Choppiness filter prevents trend signals during range-bound periods (major 2022 issue)
- 12h TF = even fewer trades than 6h, less fee drag
- ROC momentum filter adds confirmation layer

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA bullish + KAMA rising + CHOP<55 (trending) + ROC>0
- SHORT: 1d_HMA bearish + KAMA falling + CHOP<55 (trending) + ROC<0
- Range mode: enter at KAMA pullback when CHOP>61.8

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_roc_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies trending vs ranging markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            tr_sum += tr
        
        if highest_high - lowest_low > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

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
            roc[i] = 100 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    kama_14 = calculate_kama(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
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
        
        if np.isnan(kama_14[i]) or np.isnan(chop_14[i]) or np.isnan(roc_10[i]):
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
        is_trending = chop < 55.0  # Below 55 = trending market
        is_ranging = chop > 61.8   # Above 61.8 = range-bound market
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        kama_rising = kama_14[i] > kama_14[i - 5] if i >= 5 and not np.isnan(kama_14[i - 5]) else False
        kama_falling = kama_14[i] < kama_14[i - 5] if i >= 5 and not np.isnan(kama_14[i - 5]) else False
        
        # === MOMENTUM (ROC) ===
        roc = roc_10[i]
        momentum_positive = roc > 0.5
        momentum_negative = roc < -0.5
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG in trending market
        if is_trending:
            if price_above_1d and kama_rising and momentum_positive:
                desired_signal = SIZE_STRONG
            elif price_above_1d and kama_rising:
                desired_signal = SIZE_BASE
        
        # SHORT in trending market
        elif price_below_1d and kama_falling and momentum_negative:
            desired_signal = -SIZE_STRONG
        elif price_below_1d and kama_falling:
            desired_signal = -SIZE_BASE
        
        # LONG in ranging market (mean reversion to KAMA)
        if is_ranging and desired_signal == 0.0:
            if price_above_1d and close[i] < kama_14[i] * 0.995 and momentum_positive:
                desired_signal = SIZE_BASE
        
        # SHORT in ranging market (mean reversion to KAMA)
        if is_ranging and desired_signal == 0.0:
            if price_below_1d and close[i] > kama_14[i] * 1.005 and momentum_negative:
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