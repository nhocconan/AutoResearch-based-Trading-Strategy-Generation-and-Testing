#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and trend filter
# Uses Alligator's jaw/teeth/lips to identify trends in both bull and bear markets.
# Volume > 1.5x median ensures institutional participation.
# ADX(14) > 25 on 1d filters for trending markets, avoiding whipsaws in ranging conditions.
# Designed for trend following with conservative sizing to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h (jaw=13, teeth=8, lips=5, all shifted)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean()
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    
    adx_values = adx.values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 1d volume confirmation
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median()
    vol_threshold_1d = 1.5 * vol_median_1d
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d.values)
    vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold_1d.values)
    
    # Current 12h volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_threshold[i]) or
            np.isnan(vol_threshold_aligned[i])):
            continue
        
        # Alligator alignment: teeth > lips > jaw for uptrend, reverse for downtrend
        alligator_long = teeth[i] > lips[i] and lips[i] > jaw[i]
        alligator_short = teeth[i] < lips[i] and lips[i] < jaw[i]
        
        # Trend filter: ADX > 25 on both 12h and 1d
        # Calculate 12h ADX
        if i >= 14:
            tr_12h1 = high[i:] - low[i:]
            tr_12h2 = np.abs(high[i:] - close[:-i] if i > 0 else high)
            tr_12h3 = np.abs(low[i:] - close[:-i] if i > 0 else low)
            tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
            tr_12h = np.concatenate([[np.nan] * i, tr_12h]) if i > 0 else tr_12h
            tr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
            
            dm_plus_12h = np.where((high[i:] - high[:-i] if i > 0 else 0) > (low[:-i] - low[i:] if i > 0 else 0), 
                                   np.maximum(high[i:] - high[:-i] if i > 0 else 0, 0), 0)
            dm_minus_12h = np.where((low[:-i] - low[i:] if i > 0 else 0) > (high[i:] - high[:-i] if i > 0 else 0), 
                                    np.maximum(low[:-i] - low[i:] if i > 0 else 0, 0), 0)
            dm_plus_12h = np.concatenate([[0] * i, dm_plus_12h]) if i > 0 else dm_plus_12h
            dm_minus_12h = np.concatenate([[0] * i, dm_minus_12h]) if i > 0 else dm_minus_12h
            dm_plus_12h = pd.Series(dm_plus_12h).ewm(alpha=1/14, adjust=False).mean().values
            dm_minus_12h = pd.Series(dm_minus_12h).ewm(alpha=1/14, adjust=False).mean().values
            
            di_plus_12h = 100 * dm_plus_12h / tr_12h
            di_minus_12h = 100 * dm_minus_12h / tr_12h
            dx_12h = np.abs(di_plus_12h - di_minus_12h) / (di_plus_12h + di_minus_12h) * 100
            adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False).mean().values
            adx_12h_val = adx_12h[i] if i < len(adx_12h) else 0
        else:
            adx_12h_val = 0
        
        # Volume filter: current 12h volume > threshold AND 1d volume > threshold
        vol_filter = volume[i] > vol_threshold[i] and df_1d['volume'].values[i // 288] > vol_threshold_1d.values[i // 288] if i // 288 < len(vol_threshold_1d.values) else False
        
        # Long: Alligator aligned up + ADX > 25 + volume filter
        if (alligator_long and 
            adx_14_aligned[i] > 25 and adx_12h_val > 25 and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Alligator aligned down + ADX > 25 + volume filter
        elif (alligator_short and 
              adx_14_aligned[i] > 25 and adx_12h_val > 25 and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: Alligator reverses or ADX drops below 20
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (teeth[i] < lips[i] or adx_14_aligned[i] < 20)) or
               (signals[i-1] == -0.25 and (teeth[i] > lips[i] or adx_14_aligned[i] < 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsAlligator_1dADX_VolumeFilter"
timeframe = "12h"
leverage = 1.0