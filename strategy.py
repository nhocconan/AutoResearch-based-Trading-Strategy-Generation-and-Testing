#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + 12h Price Action Filter
# - Williams Alligator (13,8,5 SMAs) on 12h to identify trend direction
# - Long when Jaw > Teeth > Lips (bullish alignment) + 1d volume spike + price above 8-period SMA
# - Short when Jaw < Teeth < Lips (bearish alignment) + 1d volume spike + price below 8-period SMA
# - Volume spike confirms institutional participation
# - Price vs SMA filter ensures we trade with short-term momentum
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume analysis
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / vol_ma_20  # Volume ratio > 1.5 indicates spike
    
    # Align 1d volume ratio to 12h timeframe
    vol_ratio_12h = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate Williams Alligator on 12h timeframe
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    median_price = (high_12h + low_12h) / 2  # Williams uses median price
    
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # Shift forward by 8
    jaw[:8] = np.nan  # Fill shifted values with NaN
    
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # Shift forward by 5
    teeth[:5] = np.nan  # Fill shifted values with NaN
    
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # Shift forward by 3
    lips[:3] = np.nan  # Fill shifted values with NaN
    
    # Calculate 8-period SMA for price action filter
    sma_8 = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if NaN in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ratio_12h[i]) or np.isnan(sma_8[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol_ratio_val = vol_ratio_12h[i]
        
        if position == 0:
            # Long entry: Bullish Alligator alignment + volume spike + price above SMA
            if (jaw[i] > teeth[i] > lips[i]) and (vol_ratio_val > 1.5) and (price > sma_8[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator alignment + volume spike + price below SMA
            elif (jaw[i] < teeth[i] < lips[i]) and (vol_ratio_val > 1.5) and (price < sma_8[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks or volume drops
            if not (jaw[i] > teeth[i] > lips[i]) or (vol_ratio_val < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks or volume drops
            if not (jaw[i] < teeth[i] < lips[i]) or (vol_ratio_val < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_PriceFilter"
timeframe = "12h"
leverage = 1.0