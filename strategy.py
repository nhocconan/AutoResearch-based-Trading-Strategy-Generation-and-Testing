#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
# Camarilla pivot breakout from daily levels with EMA trend filter and volume confirmation
# Long when price breaks above R1 with volume spike and price > daily EMA34
# Short when price breaks below S1 with volume spike and price < daily EMA34
# Exit when price returns to median or trend contradicts
# Target: 20-50 trades/year on 4h timeframe with controlled risk

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_range = high_1d - low_1d
    R1 = close_1d + camarilla_range * 1.1 / 12
    S1 = close_1d - camarilla_range * 1.1 / 12
    M = close_1d  # Median/midpoint
    
    # Calculate daily EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 4h timeframe (waits for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    M_aligned = align_htf_to_ltf(prices, df_1d, M)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(M_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 with volume spike and bullish trend
            if (close[i] > R1_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume spike and bearish trend
            elif (close[i] < S1_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to median OR trend turns bearish
            if (close[i] <= M_aligned[i]) or (close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to median OR trend turns bullish
            if (close[i] >= M_aligned[i]) or (close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals