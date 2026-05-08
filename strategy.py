#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX trend filter and volume confirmation.
# Long when price breaks above R3 on 4h ADX > 25 and volume > 1.5x 20-period average.
# Short when price breaks below S3 on 4h ADX > 25 and volume > 1.5x 20-period average.
# Exit when price returns to Pivot Point (PP) or ADX drops below 20.
# Uses Camarilla pivot levels for institutional reference points and ADX to filter trend strength.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe with controlled risk.

name = "1h_Camarilla_R3S3_4hADX_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    plus_dm = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def smooth(val, period):
        result = np.full_like(val, np.nan)
        if len(val) >= period:
            result[period-1] = np.nansum(val[:period])
            for i in range(period, len(val)):
                result[i] = result[i-1] - (result[i-1] / period) + val[i]
        return result
    
    atr = smooth(tr, 14)
    plus_di = smooth(plus_dm, 14)
    minus_di = smooth(minus_dm, 14)
    
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth(dx, 14)
    adx_4h = adx
    
    # Align 4h ADX to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate Camarilla pivot levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (using previous day's data)
    pp = (high_1d + low_1d + close_1d) / 3
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align 1d Camarilla levels to 1h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, ADX > 25, volume spike
            long_cond = (close[i] > r3_aligned[i]) and (adx_4h_aligned[i] > 25) and volume_filter[i]
            # Short conditions: price breaks below S3, ADX > 25, volume spike
            short_cond = (close[i] < s3_aligned[i]) and (adx_4h_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to PP or ADX drops below 20
            if (close[i] <= pp_aligned[i]) or (adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to PP or ADX drops below 20
            if (close[i] >= pp_aligned[i]) or (adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals