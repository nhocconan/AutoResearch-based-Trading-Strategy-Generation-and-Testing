#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 Breakout with Volume Confirmation and Daily Trend Filter
# Uses 12h as primary timeframe with daily pivot levels and 1d EMA34 trend filter
# Long when: price breaks above R1 with volume > 1.5x average and above 1d EMA34
# Short when: price breaks below S1 with volume > 1.5x average and below 1d EMA34
# Camarilla levels provide institutional support/resistance; volume confirms participation
# Target: 15-35 trades/year per symbol (~60-140 total over 4 years)

name = "12h_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_1d = high_1d - low_1d
    # R1 = C + (H-L) * 1.1/12
    r1 = close_1d + range_1d * 1.1 / 12
    # S1 = C - (H-L) * 1.1/12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above R1 with volume and above 1d EMA34
            if price > r1_val and volume_confirmed and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume and below 1d EMA34
            elif price < s1_val and volume_confirmed and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to pivot or below 1d EMA34
            if price < pivot_aligned[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to pivot or above 1d EMA34
            if price > pivot_aligned[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals