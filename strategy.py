#!/usr/bin/env python3

# 6h_1d_vwap_std_dev_reversion
# Hypothesis: Mean reversion to daily VWAP with standard deviation bands on 6h timeframe
# Uses daily VWAP as fair value and 6h price deviations beyond 2 standard deviations
# Works in bull/bear markets by fading extremes and returning to value area
# Target: 25-35 trades/year (100-140 total over 4 years) with low frequency to minimize fee drag

name = "6h_1d_vwap_std_dev_reversion"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(typical_price_1d, np.nan), 
                        where=vwap_denominator!=0)
    
    # Calculate daily standard deviation of price from VWAP
    price_dev = close_1d - vwap_1d
    # Rolling 20-day std dev of price deviation
    price_dev_series = pd.Series(price_dev)
    vwap_std = price_dev_series.rolling(window=20, min_periods=20).std().values
    
    # Calculate upper and lower bands (VWAP ± 2*std)
    upper_band_1d = vwap_1d + (2.0 * vwap_std)
    lower_band_1d = vwap_1d - (2.0 * vwap_std)
    
    # Align VWAP and bands to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band_1d)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band_1d)
    
    # Volume filter: 6h volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches or goes below lower band with volume confirmation
        if (close[i] <= lower_band_aligned[i] and vol_filter[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price touches or goes above upper band with volume confirmation
        elif (close[i] >= upper_band_aligned[i] and vol_filter[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to VWAP (mean reversion complete)
        elif position == 1 and close[i] >= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals