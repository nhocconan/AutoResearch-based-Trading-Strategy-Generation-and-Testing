#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR-based volatility filter + volume confirmation
# Donchian breakouts capture strong momentum moves; 1d ATR filter ensures we trade in sufficient volatility regimes
# Volume confirmation validates breakout authenticity with institutional participation
# Works in bull/bear markets: volatility expansion occurs in both regimes, breakouts with volume are reliable
# Target: 75-150 total trades over 4 years (19-37/year) with discrete sizing 0.25

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    # ATR calculation using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.full(len(close_1d), np.nan)
    if len(tr) >= 14:
        # Initial ATR as simple average of first 14 TR values
        atr_1d[13] = np.nanmean(tr[1:15])  # Skip first NaN TR
        # Wilder's smoothing: ATR[t] = (ATR[t-1] * 13 + TR[t]) / 14
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d ATR to 4h timeframe (wait for daily close)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_width = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
            donchian_width[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
            donchian_width[i] = donchian_high[i] - donchian_low[i]
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR must be above its 50-period median (adaptive threshold)
        # Calculate rolling median of ATR for adaptive threshold
        if i >= 50:
            atr_window = atr_1d_aligned[max(0, i-49):i+1]
            atr_median = np.nanmedian(atr_window)
            vol_filter = atr_1d_aligned[i] > atr_median
        else:
            vol_filter = True  # No filter during warmup
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR volatility collapses
            if close[i] < donchian_low[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR volatility collapses
            if close[i] > donchian_high[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, volatility filter, and Donchian breakout
            if volume_confirmed and vol_filter:
                # Long entry: price > Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals