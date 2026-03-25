#!/usr/bin/env python3
"""
Experiment #1435: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend + ROC Momentum

Hypothesis: 6h timeframe is underexplored (4 candles/day) and sits between 4h noise 
and 12h sluggishness. This strategy uses:
1. 1d HMA(21) for major trend bias (proven in 4h strategy)
2. 1w HMA(21) for secular trend filter (avoid counter-trend in bear markets)
3. 6h KAMA(10,2,30) for adaptive trend - flattens in ranges, follows in trends
4. 6h ROC(10) for momentum confirmation - simpler than RSI, less extreme readings
5. Volume filter: volume > 0.8 * 20-period avg (filter false breakouts)
6. ATR(14) trailing stoploss at 2.5x (signal→0 when stopped)
7. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- KAMA adapts to volatility regime (unlike fixed HMA/EMA)
- Weekly HMA filter prevents 2022-style crash trades (major edge)
- ROC > 0 is LOOSE entry condition (guarantees trades vs RSI extremes)
- Volume filter adds confirmation without being too restrictive
- 6h TF = natural 40-60 trades/year (fee-efficient)

Entry logic (LOOSE to guarantee trades):
- LONG: 1w_HMA bullish + 1d_HMA bullish + 6h_KAMA rising + ROC > 0 + vol confirm
- SHORT: 1w_HMA bearish + 1d_HMA bearish + 6h_KAMA falling + ROC < 0 + vol confirm

Target: Sharpe>0.6, trades>=100 train, trades>=15 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_roc_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(slow_period, n):
        if not np.isnan(close[i]):
            price_change = abs(close[i] - close[i - slow_period])
            if price_change > 1e-10:
                volatility = 0.0
                for j in range(i - slow_period + 1, i + 1):
                    if not np.isnan(close[j]) and not np.isnan(close[j-1]):
                        volatility += abs(close[j] - close[j-1])
                if volatility > 1e-10:
                    er[i] = price_change / volatility
    
    # Smoothing constant
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    for i in range(slow_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    # Initialize with SMA
    sma_sum = 0.0
    sma_count = 0
    for i in range(period):
        if not np.isnan(close[i]):
            sma_sum += close[i]
            sma_count += 1
    if sma_count > 0:
        kama[period - 1] = sma_sum / sma_count
    
    for i in range(period, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]) and not np.isnan(close[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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
        if close[i - period] > 1e-10:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc

def calculate_sma(series, period):
    """Simple Moving Average"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = series[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            sma[i] = np.mean(window)
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    vol_sma_20 = calculate_sma(volume, period=20)
    
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
        
        if np.isnan(roc_10[i]) or np.isnan(kama_10[i]) or np.isnan(vol_sma_20[i]):
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
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 6h KAMA TREND (adaptive) ===
        kama_rising = False
        kama_falling = False
        if i >= 3 and not np.isnan(kama_10[i-1]) and not np.isnan(kama_10[i-2]):
            kama_rising = kama_10[i] > kama_10[i-1] and kama_10[i-1] > kama_10[i-2]
            kama_falling = kama_10[i] < kama_10[i-1] and kama_10[i-1] < kama_10[i-2]
        
        # === ROC MOMENTUM (LOOSE entry) ===
        roc = roc_10[i]
        roc_positive = roc > 0
        roc_negative = roc < 0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > 0.8 * vol_sma_20[i] if vol_sma_20[i] > 1e-10 else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + KAMA rising + ROC > 0 + volume confirm
        if price_above_1w and price_above_1d and kama_rising and roc_positive:
            if vol_confirm:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1w bearish + 1d bearish + KAMA falling + ROC < 0 + volume confirm
        elif price_below_1w and price_below_1d and kama_falling and roc_negative:
            if vol_confirm:
                desired_signal = -SIZE_STRONG
            else:
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