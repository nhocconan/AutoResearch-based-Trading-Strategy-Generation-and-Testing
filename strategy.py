#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Oversold/Overbought with 1d ADX Trend Filter and Volume Spike.
Long when Williams %R < -80 (oversold) + ADX > 25 (trending) + volume > 1.5x average.
Short when Williams %R > -20 (overbought) + ADX > 25 (trending) + volume > 1.5x average.
Exit when Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weak).
Uses 12h for price/Williams %R/volume, 1d for ADX. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on 12h
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        williams_r = np.full_like(close, -50.0)  # neutral start
        
        for i in range(period, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
            if highest_high[i] - lowest_low[i] != 0:
                williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
            else:
                williams_r[i] = -50.0
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate 14-period ADX on 1d
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        def wilders_smoothing(data, period):
            result = np.zeros_like(data)
            result[period] = np.mean(data[1:period+1])
            for i in range(period+1, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        
        dx = np.zeros_like(close)
        for i in range(len(close)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
            else:
                dx[i] = 0
        
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        adx = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + ADX trending (> 25) + volume spike
            if wr < -80 and adx > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + ADX trending (> 25) + volume spike
            elif wr > -20 and adx > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or ADX < 20 (weak trend)
            if wr >= -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or ADX < 20 (weak trend)
            if wr <= -50 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_ADXTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0