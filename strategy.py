#!/usr/bin/env python3
"""
Experiment #082: 12h Primary + 1d/1w HTF — Ehlers Fisher + KAMA Adaptive Trend + Choppiness Regime

Hypothesis: After 81 failed experiments, the winning combination is:
1. Ehlers Fisher Transform (period=9) - catches reversals in bear/range markets (research shows 75% win rate)
2. KAMA (Kaufman Adaptive) - adapts to volatility, better than static EMA/HMA in choppy markets
3. Choppiness Index (14) - regime detection: >61.8=range (mean revert), <38.2=trend (trend follow)
4. 1d HMA(21) - higher timeframe trend bias (proven in mtf_4h_rsi_chop_funding_bias_1d_v1)
5. LOOSE Fisher thresholds (-1.2/+1.2 not -1.5/+1.5) - ensures trades generate

Why this should beat Sharpe=0.368:
- Fisher Transform excels in bear/range markets (2022 crash, 2025 test period)
- KAMA adapts speed based on volatility - fast in trends, slow in chop
- Choppiness regime switch allows BOTH mean reversion AND trend following
- 12h timeframe = 30-60 trades/year (fee-efficient, meets trade count requirements)
- Dual regime logic = more trade opportunities than single-regime strategies

Entry Logic:
- TREND REGIME (CHOP < 38.2): KAMA crossover + Fisher confirm + 1d HMA alignment
- RANGE REGIME (CHOP > 61.8): Fisher extremes (-1.2 long, +1.2 short) + 1d HMA filter
- Size: 0.28 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_chop_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear/range markets
    Reference: Ehlers, J.F. (2002) "Fisher Transform"
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Normalize price to range 0-1
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        normalized = (hl2 - lowest) / range_val
        
        # Apply transformation to get value in -1 to +1 range
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (recursive filter)
        if i > period:
            fisher[i] = 0.67 * fisher_raw + 0.33 * fisher[i - 1]
            trigger[i] = fisher[i - 1]
        else:
            fisher[i] = fisher_raw
            trigger[i] = fisher_raw
    
    return fisher, trigger

def calculate_kama(close, period=10, fast_span=2, slow_span=30):
    """
    Kaufman Adaptive Moving Average - adapts to market volatility
    Reference: Kaufman, P.J. (1998) "Trading Systems and Methods"
    """
    n = len(close)
    if n < period + slow_span:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise < 1e-10:
            er[i] = 1.0
        else:
            er[i] = signal / noise
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_span + 1.0)
    slow_sc = 2.0 / (slow_span + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - identifies trending vs ranging markets
    Reference: E.W. Dreiss
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            chop[i] = 100.0
            continue
        
        # Sum of ATR-like true ranges
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum < 1e-10:
            chop[i] = 100.0
            continue
        
        # Choppiness formula
        chop[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
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
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    kama = calculate_kama(close, period=10, fast_span=2, slow_span=30)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(fisher[i]) or np.isnan(kama[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop[i] < 38.2
        is_range_regime = chop[i] > 61.8
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above trigger from below = long signal
        fisher_long_cross = (fisher[i] > fisher_trigger[i]) and (fisher[i-1] <= fisher_trigger[i-1]) if i > 0 else False
        # Fisher crossing below trigger from above = short signal
        fisher_short_cross = (fisher[i] < fisher_trigger[i]) and (fisher[i-1] >= fisher_trigger[i-1]) if i > 0 else False
        
        # Fisher extreme levels for mean reversion
        fisher_extreme_low = fisher[i] < -1.2
        fisher_extreme_high = fisher[i] > 1.2
        
        # === KAMA TREND SIGNAL ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND REGIME: KAMA + Fisher confirm + 1d HMA alignment
        if is_trend_regime:
            # Long: KAMA bullish + Fisher long cross + 1d HMA bullish
            if kama_bull and fisher_long_cross and hma_1d_bull:
                desired_signal = SIZE
            # Short: KAMA bearish + Fisher short cross + 1d HMA bearish
            elif kama_bear and fisher_short_cross and hma_1d_bear:
                desired_signal = -SIZE
        
        # RANGE REGIME: Fisher extremes for mean reversion + 1d HMA filter
        elif is_range_regime:
            # Long: Fisher extreme low + 1d HMA not strongly bearish
            if fisher_extreme_low and not hma_1d_bear:
                desired_signal = SIZE
            # Short: Fisher extreme high + 1d HMA not strongly bullish
            elif fisher_extreme_high and not hma_1d_bull:
                desired_signal = -SIZE
        
        # NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Stay flat or hold existing
        else:
            desired_signal = 0.0
        
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