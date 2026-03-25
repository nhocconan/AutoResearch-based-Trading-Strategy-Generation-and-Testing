#!/usr/bin/env python3
"""
Experiment #1347: 6h Primary + 1d/1w HTF — KAMA Trend + ROC Momentum + BB Squeeze

Hypothesis: KAMA adapts to market noise better than EMA/HMA. Combined with ROC momentum
and Bollinger Band squeeze detection, this creates high-probability breakout entries.
1d/1w HMA provides regime filter to avoid counter-trend trades.

Key features:
1. KAMA(10) adaptive trend - faster in trends, slower in chop
2. ROC(10) momentum confirmation - ensures breakout has follow-through
3. BB Width < 35th percentile = volatility compression (squeeze)
4. 1d HMA(21) for major trend bias
5. 1w HMA(21) for regime context (stronger sizing when aligned)
6. ATR(14) 2.5x trailing stop
7. Discrete sizing (0.0, ±0.25, ±0.30)

Why this should work:
- KAMA proven in baseline (Sharpe=0.447) - build on what works
- ROC momentum filters false breakouts (unlike pure Donchian)
- BB squeeze = coiling spring, breakout has energy
- 1d/1w trend filter = avoids whipsaw in counter-trend
- 6h TF = natural 30-60 trades/year (fee-friendly)
- Different from failed CHOP/CRSI mean reversion strategies

Entry logic:
- LONG: KAMA rising + ROC(10)>5 + BB squeeze + price>1d_HMA
- SHORT: KAMA falling + ROC(10)<-5 + BB squeeze + price<1d_HMA

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_roc_bb_squeeze_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(slow_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - slow_period]):
            signal = abs(close[i] - close[i - slow_period])
            noise = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
            if noise > 0:
                er[i] = signal / noise
            else:
                er[i] = 0.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(slow_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    # Calculate KAMA
    for i in range(slow_period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc

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

def calculate_bb_width(close, period=20, std_mult=2.0):
    """Bollinger Band Width - measures volatility compression"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    
    return width

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate BB Width percentile rank over lookback period"""
    n = len(bb_width)
    percentile = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid <= bb_width[i])
            percentile[i] = rank / len(valid) * 100
    
    return percentile

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    roc_10 = calculate_roc(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_width = calculate_bb_width(close, period=20, std_mult=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    
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
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(roc_10[i]):
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === KAMA TREND DIRECTION ===
        kama_rising = kama_10[i] > kama_10[i-1] if i > 0 else False
        kama_falling = kama_10[i] < kama_10[i-1] if i > 0 else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # === MOMENTUM (ROC) ===
        roc = roc_10[i]
        momentum_long = roc > 3.0  # Positive momentum
        momentum_short = roc < -3.0  # Negative momentum
        
        # === VOLATILITY SQUEEZE ===
        squeeze_detected = bb_width_pct[i] < 35  # Bottom 35% = compression
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for major regime
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: KAMA rising + momentum + squeeze + 1d trend
        if kama_rising and price_above_kama and momentum_long and squeeze_detected and price_above_1d:
            if price_above_1w:
                desired_signal = SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = SIZE_BASE  # Basic long
        
        # SHORT: KAMA falling + momentum + squeeze + 1d trend
        elif kama_falling and price_below_kama and momentum_short and squeeze_detected and price_below_1d:
            if price_below_1w:
                desired_signal = -SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = -SIZE_BASE  # Basic short
        
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