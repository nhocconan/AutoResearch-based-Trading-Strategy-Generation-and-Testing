#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h/1d Camarilla R1/S1 breakout + volume confirmation + ADX trend filter.
Long when price breaks above 4h Camarilla R1 with volume confirmation and 4h ADX > 25 (trending up).
Short when price breaks below 4h Camarilla S1 with volume confirmation and 1d ADX > 25 (trending down).
Exit when price returns to the 4h Camarilla midpoint (H4/L4) or reverses with volume.
Uses 4h/1d for structure and trend direction, 1h only for entry timing precision.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Target: 15-35 trades/year per symbol (60-140 over 4 years) to minimize fee drag.
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
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on prior 4h bar)
    range_4h = high_4h - low_4h
    r1_4h = close_4h + 0.833 * range_4h
    s1_4h = close_4h - 0.833 * range_4h
    midpoint_4h = close_4h  # Camarilla midpoint is close
    
    # Calculate 4h ADX for trend filter (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original arrays
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value: simple average
            if period < len(data):
                result[period-1] = np.nanmean(data[:period])
            # Rest: EMA
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                else:
                    result[i] = np.nan
            return result
        
        tr_smooth = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        # ADX = WilderSmooth of DX
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Get 1d data for additional trend confirmation (ADX > 20)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Align 1d ADX to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(midpoint_4h_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R1 with volume and 4h ADX > 25 (strong uptrend)
            if (close[i] > r1_4h_aligned[i] and 
                volume_confirmed and 
                adx_4h_aligned[i] > 25):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S1 with volume and 1d ADX > 20 (strong downtrend bias)
            elif (close[i] < s1_4h_aligned[i] and 
                  volume_confirmed and 
                  adx_1d_aligned[i] > 20):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S1 with volume (reversal)
            if (close[i] <= midpoint_4h_aligned[i] or 
                (close[i] < s1_4h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R1 with volume (reversal)
            if (close[i] >= midpoint_4h_aligned[i] or 
                (close[i] > r1_4h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1dCamarilla_R1S1_Breakout_Volume_ADXFilter"
timeframe = "1h"
leverage = 1.0