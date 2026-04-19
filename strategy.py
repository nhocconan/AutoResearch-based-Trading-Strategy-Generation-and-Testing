#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot R1/S1 Breakout with Volume Confirmation and 12h EMA34 Trend Filter
# Uses 6h as primary timeframe with 12h trend filter and volume confirmation
# Long when: price breaks above R1 with volume > 1.5x 20-period average and above 12h EMA34
# Short when: price breaks below S1 with volume > 1.5x 20-period average and below 12h EMA34
# Camarilla levels calculated from prior 12h bar: 
#   R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
# Volume confirmation ensures institutional participation in breakouts
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years)

name = "6h_Camarilla_R1_S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla R1 and S1 from prior 12h bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_12h - low_12h
    r1_level = close_12h + 1.1 * camarilla_range / 12.0
    s1_level = close_12h - 1.1 * camarilla_range / 12.0
    
    # Align R1/S1 to 6h timeframe (values from prior completed 12h bar)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_level)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above R1 with volume confirmation and above 12h EMA34
            if price > r1 and volume_confirmed and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume confirmation and below 12h EMA34
            elif price < s1 and volume_confirmed and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price falls back below R1 or below 12h EMA34
            if price < r1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price rises back above S1 or above 12h EMA34
            if price > s1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals