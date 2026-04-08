#!/usr/bin/env python3
# 1h_4h1d_trend_volume_v2
# Hypothesis: Combines 4h Supertrend for trend direction, 1d EMA200 for long-term bias, and 1h volume spike for entry timing. 
# Uses 1h timeframe with strict entry conditions to target 15-35 trades/year. Works in bull/bear via trend-following with volume confirmation.
# Volume spike (1.5x average) filters low-probability entries. Session filter (08-20 UTC) reduces noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_volume_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1-day data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate ATR for Supertrend (10-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10  # Wilder's smoothing
    
    # Supertrend parameters
    factor = 3.0
    upperband = (high_4h + low_4h) / 2 + factor * atr
    lowerband = (high_4h + low_4h) / 2 - factor * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_4h)
    uptrend = np.ones_like(close_4h, dtype=bool)
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upperband[i-1]:
            uptrend[i] = True
        elif close_4h[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if uptrend[i]:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # 1-day EMA200
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1h volume average (20-period)
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(len(volume)):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = vol_sum / 20
    
    # Volume spike condition (1.5x average)
    volume_spike = volume > (vol_ma * 1.5)
    
    # Align 4h Supertrend and 1d EMA200 to 1h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike.astype(float)) > 0.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure EMA200 and volume MA are ready
    
    # Pre-compute session hours for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        if np.isnan(supertrend_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend alignment filter
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Supertrend or below EMA200
            if close[i] < supertrend_aligned[i] or not price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Supertrend or above EMA200
            if close[i] > supertrend_aligned[i] or not price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price closes above Supertrend, above EMA200, and volume spike
            if close[i] > supertrend_aligned[i] and price_above_ema and volume_spike_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: price closes below Supertrend, below EMA200, and volume spike
            elif close[i] < supertrend_aligned[i] and price_below_ema and volume_spike_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals