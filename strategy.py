#!/usr/bin/env python3
# 1d_1w_supertrend_volume_v1
# Hypothesis: Trade Supertrend signals on daily timeframe with weekly trend filter and volume confirmation.
# Enter long when price closes above Supertrend (10,3) on 1d with 1w Supertrend bullish and volume > 1.5x 20-day average.
# Enter short when price closes below Supertrend (10,3) on 1d with 1w Supertrend bearish and volume > 1.5x 20-day average.
# Exit when price crosses Supertrend in opposite direction.
# Weekly trend filter prevents counter-trend trades. Volume confirms breakout strength.
# Target: 15-25 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_supertrend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Supertrend (10, 3)
    atr_period = 10
    atr_multiplier = 3
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # ATR
    atr = np.full_like(tr, np.nan)
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    # Final Supertrend
    supertrend = np.full_like(close, np.nan)
    trend = np.full_like(close, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
            
        # Final bands
        if close[i-1] <= supertrend[i-1]:
            final_upper = upper_band[i]
        else:
            final_upper = min(upper_band[i], upper_band[i-1])
            
        if close[i-1] >= supertrend[i-1]:
            final_lower = lower_band[i]
        else:
            final_lower = max(lower_band[i], lower_band[i-1])
        
        # Supertrend logic
        if close[i] > final_upper:
            supertrend[i] = final_lower
            trend[i] = -1
        elif close[i] < final_lower:
            supertrend[i] = final_upper
            trend[i] = 1
        else:
            supertrend[i] = supertrend[i-1]
            trend[i] = trend[i-1]
            
            if trend[i] == -1 and supertrend[i] > final_upper:
                supertrend[i] = final_upper
            if trend[i] == 1 and supertrend[i] < final_lower:
                supertrend[i] = final_lower
    
    # 1w Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (10, 3)
    tr1w = high_1w[1:] - low_1w[1:]
    tr2w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3w = np.abs(low_1w[1:] - close_1w[:-1])
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    trw = np.concatenate([[np.nan], trw])
    
    atrw = np.full_like(trw, np.nan)
    for i in range(atr_period, len(trw)):
        if i == atr_period:
            atrw[i] = np.nanmean(trw[i-atr_period+1:i+1])
        else:
            atrw[i] = (atrw[i-1] * (atr_period - 1) + trw[i]) / atr_period
    
    hl2w = (high_1w + low_1w) / 2
    upper_bandw = hl2w + (atr_multiplier * atrw)
    lower_bandw = hl2w - (atr_multiplier * atrw)
    
    supertrendw = np.full_like(close_1w, np.nan)
    trendw = np.full_like(close_1w, 1)
    
    for i in range(1, len(close_1w)):
        if np.isnan(upper_bandw[i-1]) or np.isnan(lower_bandw[i-1]):
            continue
            
        if close_1w[i-1] <= supertrendw[i-1]:
            final_upperw = upper_bandw[i]
        else:
            final_upperw = min(upper_bandw[i], upper_bandw[i-1])
            
        if close_1w[i-1] >= supertrendw[i-1]:
            final_lowerw = lower_bandw[i]
        else:
            final_lowerw = max(lower_bandw[i], lower_bandw[i-1])
        
        if close_1w[i] > final_upperw:
            supertrendw[i] = final_lowerw
            trendw[i] = -1
        elif close_1w[i] < final_lowerw:
            supertrendw[i] = final_upperw
            trendw[i] = 1
        else:
            supertrendw[i] = supertrendw[i-1]
            trendw[i] = trendw[i-1]
            
            if trendw[i] == -1 and supertrendw[i] > final_upperw:
                supertrendw[i] = final_upperw
            if trendw[i] == 1 and supertrendw[i] < final_lowerw:
                supertrendw[i] = final_lowerw
    
    # Align 1w Supertrend trend to 1d
    supertrendw_aligned = align_htf_to_ltf(prices, df_1w, trendw)
    
    # Volume confirmation: 1d volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 40  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend[i]) or np.isnan(supertrendw_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price closes below Supertrend
            if close[i] < supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above Supertrend
            if close[i] > supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price closes above Supertrend with bullish weekly trend and volume surge
            if (close[i] > supertrend[i] and  
                supertrendw_aligned[i] == 1 and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price closes below Supertrend with bearish weekly trend and volume surge
            elif (close[i] < supertrend[i] and 
                  supertrendw_aligned[i] == -1 and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals