#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d VWAP trend filter and volume spike
# Long when price breaks above R4 with price above 1d VWAP and volume > 2x average
# Short when price breaks below S4 with price below 1d VWAP and volume > 2x average
# Exit when price retouches the central pivot (PP)
# R4/S4 are less commonly used than R3/S3, offering stronger breakout signals with fewer trades
# Designed to capture significant breakouts with institutional levels, VWAP for trend, volume for conviction
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_Camarilla_R4S4_Breakout_1dVWAP_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d VWAP for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Typical price and cumulative VWAP calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap = vwap.values
    
    # Align VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Calculate 1d OHLC for Camarilla levels (using previous day)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate R4 and S4 levels (extended Camarilla)
    r4 = pp + (prev_high - prev_low) * 1.5000
    s4 = pp - (prev_high - prev_low) * 1.5000
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for VWAP calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vwap_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R4, price above VWAP, volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > vwap_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S4, price below VWAP, volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < vwap_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches central pivot
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches central pivot
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals