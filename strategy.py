#!/usr/bin/env python3
"""
12h_1d_Adaptive_Channel_Breakout_v1
Hypothesis: 12h Donchian channel (20) breakouts with 1d volatility filter and volume confirmation.
The strategy adapts to volatility regimes: in high volatility (expanding channel), it follows breakouts;
in low volatility (contracting channel), it fades reversals at channel edges. This works in both bull
and bear markets by capturing momentum during trends and mean reversion during ranges. Uses 1d for
volatility regime filter to avoid whipsaws. Target: 15-25 trades per year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Adaptive_Channel_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12H DONCHIAN CHANNEL (20-period) ===
    donchian_len = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    if n >= donchian_len:
        # Calculate rolling max/min
        for i in range(donchian_len - 1, n):
            highest_high[i] = np.max(high[i - donchian_len + 1:i + 1])
            lowest_low[i] = np.min(low[i - donchian_len + 1:i + 1])
    
    # === 1D VOLATILITY REGIME (ATR ratio) ===
    # Calculate ATR(10) and ATR(30) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # ATR(10) and ATR(30)
    atr10 = np.full(len(tr), np.nan)
    atr30 = np.full(len(tr), np.nan)
    
    if len(tr) >= 30:
        # Initial ATR values
        atr10[9] = np.nanmean(tr[1:11])  # Skip first NaN
        atr30[29] = np.nanmean(tr[1:31])
        
        # Wilder's smoothing
        for i in range(10, len(tr)):
            if not np.isnan(atr10[i-1]):
                atr10[i] = (atr10[i-1] * 9 + tr[i]) / 10
        for i in range(30, len(tr)):
            if not np.isnan(atr30[i-1]):
                atr30[i] = (atr30[i-1] * 29 + tr[i]) / 30
    
    # Avoid division by zero
    atr_ratio = np.where(atr30 > 0, atr10 / atr30, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === 12H VOLUME CONFIRMATION ===
    vol_ma = np.full(n, np.nan)
    vol_len = 20
    if n >= vol_len:
        vol_sum = np.sum(volume[:vol_len])
        vol_ma[vol_len-1] = vol_sum / vol_len
        for i in range(vol_len, n):
            vol_sum = vol_sum - volume[i-vol_len] + volume[i]
            vol_ma[i] = vol_sum / vol_len
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility regime: high vol (expanding) > 0.8, low vol (contracting) <= 0.8
        high_vol_regime = atr_ratio_aligned[i] > 0.8
        
        # Channel width for sensitivity
        channel_width = (highest_high[i] - lowest_low[i]) / closest_low[i] if lowest_low[i] > 0 else 0
        
        # Adaptive logic based on regime
        if high_vol_regime:
            # HIGH VOLATILITY: Follow breakouts (momentum)
            long_breakout = high[i] > highest_high[i]
            short_breakout = low[i] < lowest_low[i]
            
            # Entry with volume confirmation
            if long_breakout and vol_spike[i] and position != 1:
                position = 1
                signals[i] = 0.30
            elif short_breakout and vol_spike[i] and position != -1:
                position = -1
                signals[i] = -0.30
            
            # Exit when price returns to middle of channel
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if position == 1 and close[i] <= mid_channel:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] >= mid_channel:
                position = 0
                signals[i] = 0.0
                
        else:
            # LOW VOLATILITY: Fade reversals at channel edges (mean reversion)
            # Define entry zones near channel edges
            upper_zone = lowest_low[i] + channel_width * 0.85
            lower_zone = lowest_low[i] + channel_width * 0.15
            
            # Long when price rejects lower edge
            long_setup = (low[i] <= lowest_low[i] * 1.002) and (close[i] > lower_zone)
            # Short when price rejects upper edge
            short_setup = (high[i] >= highest_high[i] * 0.998) and (close[i] < upper_zone)
            
            # Entry with volume confirmation (avoid fakeouts)
            if long_setup and vol_spike[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_setup and vol_spike[i] and position != -1:
                position = -1
                signals[i] = -0.25
            
            # Exit when price reaches opposite zone or middle
            if position == 1 and close[i] >= upper_zone:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] <= lower_zone:
                position = 0
                signals[i] = 0.0
    
    return signals