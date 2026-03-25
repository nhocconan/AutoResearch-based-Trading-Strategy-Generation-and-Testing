#!/usr/bin/env python3
"""
Experiment #1443: 6h Primary + 1d/1w HTF — Adaptive KAMA Trend + Volume Confirmation

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy combines:
1. 1w ROC(10) for major weekly momentum bias (avoid fighting strong weekly trends)
2. 1d HMA(21) for daily trend direction
3. 6h KAMA(14) adaptive MA for trend following (adjusts to volatility)
4. 6h Volume spike confirmation (vol > 1.5x 20-period avg) to reduce false breakouts
5. ATR(14) trailing stoploss (signal→0 when stopped)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work where other 6h strategies failed:
- KAMA adapts to volatility better than HMA/EMA (less whipsaw in chop)
- Volume confirmation filters out low-conviction moves
- 1w ROC filter prevents counter-trend trades during strong weekly momentum
- LOOSE entry conditions: trend alignment + volume spike (not extreme RSI etc.)
- 6h TF = natural 25-45 trades/year (fee-efficient middle ground)

Entry logic (LOOSE to guarantee trades):
- LONG: 1w_ROC > 0 + 1d_HMA bullish + 6h_KAMA rising + volume > 1.5x avg
- SHORT: 1w_ROC < 0 + 1d_HMA bearish + 6h_KAMA falling + volume > 1.5x avg

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_volume_roc_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market volatility"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 1e-10:
                er[i] = signal / noise
            else:
                er[i] = 1.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            if not np.isnan(kama[i - 1]):
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
    """Rate of Change"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0 and not np.isnan(close[i]) and not np.isnan(close[i - period]):
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100.0
    
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

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

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
    
    roc_1w_raw = calculate_roc(df_1w['close'].values, period=10)
    roc_1w_aligned = align_htf_to_ltf(prices, df_1w, roc_1w_raw)
    
    # Calculate 6h indicators
    kama_14 = calculate_kama(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
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
        
        if np.isnan(kama_14[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(roc_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY MOMENTUM BIAS (1w ROC) ===
        weekly_bullish = roc_1w_aligned[i] > 0
        weekly_bearish = roc_1w_aligned[i] < 0
        
        # === DAILY TREND DIRECTION (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 6h KAMA TREND (adaptive) ===
        kama_rising = False
        kama_falling = False
        if i >= 3 and not np.isnan(kama_14[i-1]) and not np.isnan(kama_14[i-2]):
            kama_rising = kama_14[i] > kama_14[i-1] and kama_14[i-1] > kama_14[i-2]
            kama_falling = kama_14[i] < kama_14[i-1] and kama_14[i-1] < kama_14[i-2]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + 6h KAMA rising + volume spike
        if weekly_bullish and price_above_1d and kama_rising and vol_spike:
            desired_signal = SIZE_STRONG
        
        # SHORT: 1w bearish + 1d bearish + 6h KAMA falling + volume spike
        elif weekly_bearish and price_below_1d and kama_falling and vol_spike:
            desired_signal = -SIZE_STRONG
        
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