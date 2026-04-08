# 6h_1w_1d_volume_accumulation_distribution_v1
# Hypothesis: Trade based on accumulation/distribution line from weekly timeframe combined with 1d trend and volume confirmation on 6h.
# Uses weekly Chaikin Money Flow (CMF) to detect institutional accumulation/distribution, filtered by 1d EMA trend.
# Long when weekly CMF > 0.1 and price > 1d EMA50 with volume > 1.5x average.
# Short when weekly CMF < -0.1 and price < 1d EMA50 with volume > 1.5x average.
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Weekly CMF filters out noise and captures smart money flow, working in both bull and bear markets by following institutional activity.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_volume_accumulation_distribution_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for CMF calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Money Flow Multiplier and Volume for weekly CMF
    # Avoid division by zero when high == low
    hl_range = high_1w - low_1w
    mf_multiplier = np.where(hl_range != 0, ((close_1w - low_1w) - (high_1w - close_1w)) / hl_range, 0.0)
    mf_volume = mf_multiplier * volume_1w
    
    # Calculate CMF(20) - 20-period Chaikin Money Flow
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume_1w).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(volume_sum != 0, mf_volume_sum / volume_sum, 0.0)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 for daily trend
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly CMF and daily EMA50 to 6h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1w, cmf)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure EMA50 and CMF are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_50[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_50[i] if vol_ma_50[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: weekly CMF turns negative or price breaks below daily EMA50
            if cmf_aligned[i] < 0 or close[i] < ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly CMF turns positive or price breaks above daily EMA50
            if cmf_aligned[i] > 0 or close[i] > ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: weekly CMF > 0.1 (accumulation) and price > daily EMA50 with volume surge
            if (cmf_aligned[i] > 0.1 and close[i] > ema50_aligned[i] and vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: weekly CMF < -0.1 (distribution) and price < daily EMA50 with volume surge
            elif (cmf_aligned[i] < -0.1 and close[i] < ema50_aligned[i] and vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals