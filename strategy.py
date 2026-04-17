#!/usr/bin/env python3
"""
Hypothesis: 4h Trend + 1h Camarilla Breakout with Volume Spike.
- Use 4h EMA34 for trend direction (long if close > EMA34, short if close < EMA34).
- On 1h timeframe, enter long at Camarilla R1 breakout with volume spike (>1.5x 20-bar avg volume).
- Enter short at Camarilla S1 breakout with volume spike.
- Exit when price crosses Camarilla pivot point (PP) or 4h trend reverses.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year).
- Uses 4h for trend, 1h for precise entry/exit, volume confirmation to reduce false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1h data for Camarilla levels (using daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla: PP = (H+L+C)/3, R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-period average volume for volume spike confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precompute hour array)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(avg_volume_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_4h_aligned[i]
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume_20[i]
        
        # Volume spike condition: current volume > 1.5 * 20-period average volume
        volume_spike = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price > 4h EMA34 (uptrend) AND breaks above Camarilla R1 AND volume spike
            if price > ema34 and price > r1 and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA34 (downtrend) AND breaks below Camarilla S1 AND volume spike
            elif price < ema34 and price < s1 and volume_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla PP OR 4h trend reverses (price < EMA34)
            if price < pp or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price > Camarilla PP OR 4h trend reverses (price > EMA34)
            if price > pp or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "4hTrend_1hCamarilla_VolumeSpike"
timeframe = "1h"
leverage = 1.0