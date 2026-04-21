#!/usr/bin/env python3
"""
1d_1w_TRIX_VolumeSpike_TrendFilter
Hypothesis: TRIX (TRIple Exponential Average) on daily charts identifies momentum shifts. Combined with weekly trend filter (EMA34) and volume spikes (>2x 20-day avg), it captures institutional momentum bursts. Works in bull/bear by aligning with weekly trend. Target 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(series, period):
    """Exponential Moving Average with proper initialization."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    result = np.zeros(len(series))
    alpha = 2.0 / (period + 1)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result

def trix(close, period=12):
    """TRIX indicator: percentage change of triple-smoothed EMA."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    # Calculate percentage change
    result = np.full(len(close), np.nan)
    for i in range(1, len(e3)):
        if e3[i-1] != 0:
            result[i] = (e3[i] - e3[i-1]) / e3[i-1] * 100
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = ema(close_1w, 34)
    
    # Align weekly EMA34 to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(12) on daily closes
    trix_val = trix(close, 12)
    
    # Volume filter: volume > 2.0x 20-day average (institutional participation)
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            volume_avg[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(trix_val[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        trix_now = trix_val[i]
        ema34 = ema34_1w_aligned[i]
        vol_confirm = volume_spike[i]
        
        # Simple ATR-based stoploss (20-period)
        if i >= 20:
            tr_sum = 0
            for j in range(1, 21):
                idx = i - j
                if idx >= 0:
                    tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
                    tr_sum += tr
            atr = tr_sum / 20
        else:
            atr = 0
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: TRIX turns positive with volume spike in uptrend (price > weekly EMA34)
            if trix_now > 0 and trix_val[i-1] <= 0 and vol_confirm and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TRIX turns negative with volume spike in downtrend (price < weekly EMA34)
            elif trix_now < 0 and trix_val[i-1] >= 0 and vol_confirm and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: TRIX turns negative or trend breaks
            if trix_now < 0 or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns positive or trend breaks
            if trix_now > 0 or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_TRIX_VolumeSpike_TrendFilter"
timeframe = "1d"
leverage = 1.0