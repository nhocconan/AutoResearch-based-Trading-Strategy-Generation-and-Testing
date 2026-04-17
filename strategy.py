#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d Volume Spike and ADX Trend Filter.
Long when Williams %R crosses above -80 from below AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
Short when Williams %R crosses below -20 from above AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
Exit when Williams %R crosses opposite extreme (-20 for longs, -80 for shorts) or volume condition fails.
Uses 1d for volume spike and ADX trend confirmation, 4h for Williams %R entry timing.
Target: 20-50 total trades over 4 years (5-12/year). Williams %R captures oversold/overbought conditions,
volume spike confirms institutional interest, ADX ensures trending environment to avoid chop whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend strength
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 1d volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (1.5 * vol_ma)
    
    # Calculate Williams %R on 4h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5
        
        # Williams %R crossovers
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND volume spike AND ADX > 25
            if wr > -80 and wr_prev <= -80 and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND volume spike AND ADX > 25
            elif wr < -20 and wr_prev >= -20 and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR volume spike fails OR ADX < 20
            if wr > -20 or not vol_spike or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR volume spike fails OR ADX < 20
            if wr < -80 or not vol_spike or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0