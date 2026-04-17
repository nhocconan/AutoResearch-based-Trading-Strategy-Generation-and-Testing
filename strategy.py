#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d Volume Spike + ADX Trend Filter.
Long when Williams %R < -80 (oversold) with volume > 1.8x average and ADX > 25 (trending).
Short when Williams %R > -20 (overbought) with volume > 1.8x average and ADX > 25.
Exit when Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakens).
Uses 1d for volume spike and ADX calculation, 6h for Williams %R.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures mean reversion in trends,
volume spike confirms institutional interest, ADX filter avoids choppy markets.
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
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(close)
        dm_minus = np.zeros_like(close)
        for i in range(1, len(close)):
            dm_plus[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            dm_minus[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        tr_period = np.zeros_like(close)
        dm_plus_period = np.zeros_like(close)
        dm_minus_period = np.zeros_like(close)
        
        tr_period[period] = np.mean(tr[1:period+1])
        dm_plus_period[period] = np.mean(dm_plus[1:period+1])
        dm_minus_period[period] = np.mean(dm_minus[1:period+1])
        
        for i in range(period+1, len(close)):
            tr_period[i] = (tr_period[i-1] * (period-1) + tr[i]) / period
            dm_plus_period[i] = (dm_plus_period[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_period[i] = (dm_minus_period[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(close)
        di_minus = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if tr_period[i] > 0:
                di_plus[i] = (dm_plus_period[i] / tr_period[i]) * 100
                di_minus[i] = (dm_minus_period[i] / tr_period[i]) * 100
                dx[i] = (abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
            else:
                di_plus[i] = 0
                di_minus[i] = 0
                dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate 1d Volume Spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma * 1.8)
    
    # Calculate ADX
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 6h Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        williams_r = np.zeros_like(close)
        
        for i in range(len(close)):
            if i < period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
                williams_r[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
                if highest_high[i] - lowest_low[i] != 0:
                    williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
                else:
                    williams_r[i] = -50  # neutral when no range
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Align 1d indicators to 6h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        vol_spike = volume_spike_1d_aligned[i]
        adx = adx_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume spike and ADX > 25 (trending)
            if wr < -80 and vol_spike and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume spike and ADX > 25 (trending)
            elif wr > -20 and vol_spike and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakens)
            if wr >= -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakens)
            if wr <= -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_VolumeSpike_ADXTrend"
timeframe = "6h"
leverage = 1.0