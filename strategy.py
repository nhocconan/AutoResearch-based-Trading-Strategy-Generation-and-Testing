#!/usr/bin/env python3
"""
Experiment #1212: 12h KAMA Adaptive Trend + 1d HMA + Choppiness Regime

Hypothesis: After 1200+ experiments, the pattern is clear - over-filtered entries = 0 trades.
This strategy uses ADAPTIVE trend following that works in BOTH trending and ranging markets:

1. KAMA(21) on 12h - adapts to volatility (fast in trends, slow in chop)
2. 1d HMA(21) - higher timeframe trend confirmation (loaded ONCE via mtf_data)
3. Choppiness Index(14) - regime detection but NOT required for entry (loose filter)
4. RSI(14) with WIDE bands (25-75) - triggers on normal moves, not extremes

Key insight: Previous strategies failed because they required ALL conditions to align.
This strategy uses SOFT confluence - each factor adds conviction but none blocks entry.

Entry logic (LOOSE - guarantee 30+ trades/year):
- LONG: price > KAMA AND RSI < 70 (momentum continuation, not just pullback)
- SHORT: price < KAMA AND RSI > 30 (momentum continuation)
- 1d HMA adds size conviction (0.30 if aligned, 0.25 if not)
- Choppiness adjusts sizing (reduce in extreme chop)

Why this should beat Sharpe=0.445:
- KAMA adapts better than HMA/EMA in crypto's mixed regimes
- 12h natural frequency = 20-50 trades/year (fee-optimal)
- Wide RSI bands = entries on normal moves, not rare extremes
- Discrete sizing = minimal fee churn
- ATR trailing stop = protects from 2022-style crashes

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adaptive_chop_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    Fast in trends, slow in choppy markets
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(close[i - period]):
            er[i] = 0.0
            continue
        price_change = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                volatility += abs(close[j] - close[j - 1])
        er[i] = price_change / volatility if volatility > 1e-10 else 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        if not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    Measures market choppiness vs trending
    > 61.8 = choppy/ranging, < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest or np.isnan(highest) or np.isnan(lowest):
            chop[i] = 50.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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
    kama_21 = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_REDUCED = 0.15
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (KAMA + Daily HMA) ===
        price_above_kama = close[i] > kama_21[i]
        price_below_kama = close[i] < kama_21[i]
        
        # Daily HMA for higher timeframe confirmation
        hma_1d_valid = not np.isnan(hma_1d_aligned[i])
        price_above_1d = hma_1d_valid and close[i] > hma_1d_aligned[i]
        price_below_1d = hma_1d_valid and close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness - SOFT filter) ===
        chop_valid = not np.isnan(chop_14[i])
        is_choppy = chop_valid and chop_14[i] > 55.0  # Soft chop threshold
        is_trending = chop_valid and chop_14[i] < 45.0  # Soft trend threshold
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        rsi = rsi_14[i]
        
        # LONG: Price above KAMA + RSI not overbought (< 75)
        # This allows momentum continuation, not just pullbacks
        if price_above_kama:
            if rsi < 75.0:  # Wide band - allows continuation
                # Base size
                desired_signal = SIZE_BASE
                
                # Increase size if 1d HMA confirms
                if price_above_1d:
                    desired_signal = SIZE_STRONG
                
                # Reduce size in choppy regime (but don't block entry)
                if is_choppy:
                    desired_signal = max(SIZE_REDUCED, desired_signal * 0.7)
        
        # SHORT: Price below KAMA + RSI not oversold (> 25)
        elif price_below_kama:
            if rsi > 25.0:  # Wide band - allows continuation
                # Base size
                desired_signal = -SIZE_BASE
                
                # Increase size if 1d HMA confirms
                if price_below_1d:
                    desired_signal = -SIZE_STRONG
                
                # Reduce size in choppy regime (but don't block entry)
                if is_choppy:
                    desired_signal = min(-SIZE_REDUCED, desired_signal * 0.7)
        
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
        elif desired_signal >= SIZE_REDUCED * 0.9:
            final_signal = SIZE_REDUCED
        elif desired_signal <= -SIZE_REDUCED * 0.9:
            final_signal = -SIZE_REDUCED
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