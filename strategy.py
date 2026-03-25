#!/usr/bin/env python3
"""
Experiment #1580: 6h Primary + 1d/1w HTF — Adaptive KAMA Trend with Vol Regime

Hypothesis: 6h timeframe sits in the "Goldilocks zone" between 4h (too noisy) and 12h (too slow).
This strategy uses KAMA (Kaufman Adaptive Moving Average) which adapts to market efficiency,
combined with Bollinger Band Width regime detection and 1d/1w trend confirmation.

Key components:
1. 1w HMA(21) for major secular trend bias (avoid counter-trend in strong trends)
2. 1d KAMA(10) for intermediate trend direction
3. 6h KAMA(10) + KAMA(30) for entry timing (adaptive to volatility)
4. BB Width percentile for regime: narrow = breakout likely, wide = mean revert
5. ROC(10) momentum confirmation (avoid entering against momentum)
6. ATR(14) trailing stoploss (2.5x ATR)
7. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work:
- KAMA adapts to market conditions (fast in trends, slow in chop)
- 6h TF = natural 30-50 trades/year (fee-efficient)
- LOOSE entry thresholds guarantee trades (KAMA cross + ROC confirm)
- 1w/1d HTF filter prevents major counter-trend disasters
- BB Width regime switches between breakout and pullback logic

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + 1d_KAMA bullish + 6h_KAMA10>30 + ROC>0
- SHORT: 1w_HMA bearish + 1d_KAMA bearish + 6h_KAMA10<30 + ROC<0
- Pullback entries when BB Width narrow (trend continuation)
- Breakout entries when BB Width expands (new trend)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_adaptive_trend_bbwidth_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in chop.
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio (ER) = |Net Change| / Sum of Absolute Changes
    er = np.zeros(n)
    for i in range(10, n):
        net_change = abs(close[i] - close[i - 10])
        sum_changes = np.sum(np.abs(np.diff(close[i - 10:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Smoothing Constants
    fast_sc = 2.0 / (2.0 + 1.0)  # Fast SC for period=2
    slow_sc = 2.0 / (2.0 + 30.0)  # Slow SC for period=30
    
    # Initialize KAMA
    kama[9] = close[9]  # Start at first valid price
    
    for i in range(10, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def calculate_bollinger_width(close, period=20):
    """Bollinger Band Width = (Upper - Lower) / Middle"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    width = np.full(n, np.nan, dtype=np.float64)
    mask = sma != 0
    width[mask] = (2.0 * std[mask]) / sma[mask]
    
    return width

def calculate_bb_width_percentile(bb_width, lookback=50):
    """Percentile rank of BB Width over lookback period"""
    n = len(bb_width)
    percentile = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            window = bb_width[i - lookback:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                percentile[i] = 100.0 * np.sum(valid < bb_width[i]) / len(valid)
    
    return percentile

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
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    kama_10 = calculate_kama(close, period=10)
    kama_30 = calculate_kama(close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    bb_width = calculate_bollinger_width(close, period=20)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=50)
    
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
        
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]) or np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (BB Width Percentile) ===
        bb_pct = bb_width_pct[i]
        is_squeeze_regime = bb_pct < 30.0  # Narrow bands = breakout likely
        is_expansion_regime = bb_pct > 70.0  # Wide bands = mean revert likely
        is_neutral_regime = not is_squeeze_regime and not is_expansion_regime
        
        # === TREND DIRECTION (1w HMA + 1d KAMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        kama_1d_bullish = close[i] > kama_1d_aligned[i]
        kama_1d_bearish = close[i] < kama_1d_aligned[i]
        
        # === 6h KAMA CROSSOVER (adaptive trend momentum) ===
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # === ROC MOMENTUM ===
        roc = roc_10[i]
        roc_positive = roc > 0.5  # Slight positive momentum
        roc_negative = roc < -0.5  # Slight negative momentum
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # SQUEEZE REGIME: Breakout entries (narrow BB = explosive move coming)
        if is_squeeze_regime:
            # LONG: 1w bullish + 1d bullish + KAMA cross up + ROC positive
            if price_above_1w and kama_1d_bullish and kama_bullish and roc_positive:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1w bearish + 1d bearish + KAMA cross down + ROC negative
            elif price_below_1w and kama_1d_bearish and kama_bearish and roc_negative:
                desired_signal = -SIZE_STRONG
        
        # EXPANSION REGIME: Pullback entries (wide BB = mean reversion likely)
        elif is_expansion_regime:
            # LONG: 1w bullish + KAMA bullish but price pulled back to KAMA30
            if price_above_1w and kama_bullish and close[i] < kama_10[i] * 1.002 and roc > -2.0:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + KAMA bearish but price rallied to KAMA10
            elif price_below_1w and kama_bearish and close[i] > kama_10[i] * 0.998 and roc < 2.0:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Standard trend following
        elif is_neutral_regime:
            # LONG: 1w bullish + 1d bullish + KAMA cross + ROC confirm
            if price_above_1w and kama_1d_bullish and kama_bullish and roc > 0:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + 1d bearish + KAMA cross + ROC confirm
            elif price_below_1w and kama_1d_bearish and kama_bearish and roc < 0:
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