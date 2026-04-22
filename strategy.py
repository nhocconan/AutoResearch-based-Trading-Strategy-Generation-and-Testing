#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d ADX trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions with mean-reversion tendency.
# 1d ADX > 25 ensures trades align with strong daily trends for higher probability.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Designed for 12h timeframe targeting 15-25 trades/year with strong performance in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def smooth(values, period):
            smoothed = np.zeros_like(values)
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
            return smoothed
        
        atr = smooth(tr, period)
        dm_plus_smooth = smooth(dm_plus, period)
        dm_minus_smooth = smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.nansum(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(len(high)):
            if i < period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    williams_r_14 = calculate_williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_avg_20[i] = np.nan
        else:
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(williams_r_14[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + strong uptrend (ADX > 25) + volume spike
            if (williams_r_14[i] < -80 and 
                adx_14_1d_aligned[i] > 25 and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + strong downtrend (ADX > 25) + volume spike
            elif (williams_r_14[i] > -20 and 
                  adx_14_1d_aligned[i] > 25 and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral territory (-50) or trend weakens (ADX < 20)
            if position == 1:
                # Exit long: Williams %R returns above -50 or trend weakens
                if (williams_r_14[i] > -50 or 
                    adx_14_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Williams %R returns below -50 or trend weakens
                if (williams_r_14[i] < -50 or 
                    adx_14_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dADX25_VolumeSpike"
timeframe = "12h"
leverage = 1.0