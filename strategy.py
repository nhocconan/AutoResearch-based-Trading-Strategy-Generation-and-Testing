#!/usr/bin/env python3
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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily pivot points (Camarilla style for intraday relevance)
    # Camarilla: based on previous day's range
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    daily_range = daily_high - daily_low
    camarilla_pp = (daily_high + daily_low + daily_close) / 3.0
    camarilla_r1 = camarilla_pp + (daily_range * 1.1 / 12)
    camarilla_s1 = camarilla_pp - (daily_range * 1.1 / 12)
    camarilla_r2 = camarilla_pp + (daily_range * 1.1 / 6)
    camarilla_s2 = camarilla_pp - (daily_range * 1.1 / 6)
    camarilla_r3 = camarilla_pp + (daily_range * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (daily_range * 1.1 / 4)
    camarilla_r4 = camarilla_pp + (daily_range * 1.1 / 2)
    camarilla_s4 = camarilla_pp - (daily_range * 1.1 / 2)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe with proper delay
    pp_4h = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4h Donchian channels (20-period) for structure
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(r3_4h[i]) or 
            np.isnan(s3_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: use ATR ratio to detect trending vs ranging markets
        # Low ATR ratio = ranging (mean revert at pivot)
        # High ATR ratio = trending (breakout)
        atr_ratio = atr_14_4h[i] / (0.01 * close[i])  # normalized ATR
        
        # Entry conditions with regime adaptation:
        # Ranging market (ATR ratio < 1.5): mean revert at S1/R1
        # Trending market (ATR ratio >= 1.5): breakout at R2/S2
        
        if atr_ratio < 1.5:  # Ranging market - mean reversion
            # Long conditions: price touches S1 with volume confirmation
            if (close[i] <= s1_4h[i] * 1.002 and   # Allow 0.2% tolerance for touch
                close[i] >= lowest_20[i] and       # Not breaking structure
                volume_ratio[i] > 1.2 and          # Volume confirmation
                atr_14_4h[i] > 0.003 * close[i]):  # Minimum volatility
                signals[i] = 0.25
                
            # Short conditions: price touches R1 with volume confirmation
            elif (close[i] >= r1_4h[i] * 0.998 and  # Allow 0.2% tolerance for touch
                  close[i] <= highest_20[i] and     # Not breaking structure
                  volume_ratio[i] > 1.2 and         # Volume confirmation
                  atr_14_4h[i] > 0.003 * close[i]): # Minimum volatility
                signals[i] = -0.25
        else:  # Trending market - breakout
            # Long conditions: price breaks above R2 with volume confirmation
            if (close[i] > r2_4h[i] and            # Clear breakout above R2
                volume_ratio[i] > 1.5 and          # Strong volume confirmation
                atr_14_4h[i] > 0.005 * close[i]):  # Sufficient volatility for trend
                signals[i] = 0.25
                
            # Short conditions: price breaks below S2 with volume confirmation
            elif (close[i] < s2_4h[i] and          # Clear breakdown below S2
                  volume_ratio[i] > 1.5 and        # Strong volume confirmation
                  atr_14_4h[i] > 0.005 * close[i]): # Sufficient volatility for trend
                signals[i] = -0.25
        # Default: flat
    
    return signals

name = "4h_Camarilla_Pivot_MeanRev_Breakout_Regime"
timeframe = "4h"
leverage = 1.0