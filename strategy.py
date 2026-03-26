#!/usr/bin/env python3
"""
Experiment #011: 6h Williams %R + KAMA + ATR Volatility Expansion

HYPOTHESIS: Williams %R identifies oversold/overbought extremes that precede mean 
reversion. In bear markets (2022), buying when %R<-80 AND price near daily low 
catches panic bottoms. In bull markets, shorting when %R>-20 AND price near daily 
high catches rallies. 1d KAMA filters trades to trend direction. ATR expansion 
confirms volatility spikes (high conviction setups).

Unique aspects vs. failed strategies:
- Williams %R instead of RSI (different calculation, more responsive)
- Range position (where is price within daily range) for entry timing
- 1d KAMA for trend (not SMA200 or HMA)

TIMEFRAME: 6h primary
HTF: 1d for KAMA trend alignment
TARGET: 40-80 total trades over 4 years (10-20/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_kama_atr_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_kama(close, period=10):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
       volatility = 0.0
        for j in range(i - period + 1, i):
            volatility += abs(close[j + 1] - close[j])
        if volatility > 0:
            er[i] = change / volatility
    
    # Smoothing constants
    fast = 0.666  # 2/(2+1)
    slow = 0.064  # 2/(30+1)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]  # Initialize
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast - slow) + slow) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams %R (14-period, standard)
    williams_period = 14
    williams_r = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(williams_period - 1, n):
        window_high = np.max(high[i - williams_period + 1:i + 1])
        window_low = np.min(low[i - williams_period + 1:i + 1])
        if window_high != window_low:
            williams_r[i] = -100 * (window_high - close[i]) / (window_high - window_low)
        else:
            williams_r[i] = -50
    
    # ATR expansion ratio: current ATR vs 30-bar ATR mean
    atr_mean = pd.Series(atr_14).rolling(window=30, min_periods=15).mean().values
    atr_ratio = atr_14 / np.where(atr_mean > 0, atr_mean, 1)
    
    # Range position: where is price within daily range (need 1d high/low for this)
    # Use rolling 24-bar (6h * 4 = 24 bars per day) high/low as proxy for daily
    daily_high_proxy = pd.Series(high).rolling(window=24, min_periods=12).max().values
    daily_low_proxy = pd.Series(low).rolling(window=24, min_periods=12).min().values
    daily_range = daily_high_proxy - daily_low_proxy
    
    range_position = np.where(
        daily_range > 0,
        (close - daily_low_proxy) / daily_range,
        0.5
    )
    
    # Volume spike
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    vol_spike = vol_ratio > 1.4
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60  # Williams %R needs 14 bars + other indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Get current values
        wr = williams_r[i]
        atr_ratio_val = atr_ratio[i]
        range_pos = range_position[i]
        vol_spike_val = vol_spike[i]
        
        # Daily trend: price above KAMA = bullish
        daily_bullish = close[i] > kama_1d_aligned[i]
        
        desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    stoploss_triggered = True
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        elif in_position:
            # Maintain position (don't churn)
            desired_signal = position_side * SIZE
        else:
            # === NEW ENTRY CONDITIONS ===
            
            # LONG ENTRY: Williams %R oversold (<-80) + near daily low (<0.35) + bullish 1d trend
            # Add ATR expansion for extra conviction
            long_conditions = (
                wr < -80 and          # Oversold
                range_pos < 0.35 and  # Price near daily low
                daily_bullish         # 1d trend aligned
            )
            
            if long_conditions:
                desired_signal = SIZE
            
            # SHORT ENTRY: Williams %R overbought (>-20) + near daily high (>0.65) + bearish 1d trend
            short_conditions = (
                wr > -20 and         # Overbought
                range_pos > 0.65 and # Price near daily high
                not daily_bullish    # 1d trend aligned
            )
            
            if short_conditions:
                desired_signal = -SIZE
        
        # === POSITION ENTRY ===
        if desired_signal != 0.0 and not in_position:
            in_position = True
            position_side = int(np.sign(desired_signal))
            entry_price = close[i]
            entry_atr = atr_14[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
            else:
                stop_price = entry_price + 2.5 * entry_atr
        
        # === POSITION EXIT (signal flips to 0) ===
        if desired_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            stop_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals