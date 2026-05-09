# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R4/S4 breakout with 1d VWAP trend filter and volume spike
# Long when price breaks above R4 with price > 1d VWAP and volume > 1.8x average
# Short when price breaks below S4 with price < 1d VWAP and volume > 1.8x average
# Exit when price returns to the central pivot (PP) or reverses to opposite R1/S1
# Uses 12h timeframe to reduce trade frequency, VWAP for institutional trend, volume for conviction
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "12h_Camarilla_R4S4_Breakout_1dVWAP_VolumeSpike"
timeframe = "12h"
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
    
    # Typical price and cumulative VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    cum_tpv = (typical_price * df_1d['volume']).cumsum()
    cum_vol = df_1d['volume'].cumsum()
    vwap = cum_tpv / cum_vol
    vwap_prev = vwap.shift(1)  # Previous day's VWAP for trend filter
    
    # Calculate 1d OHLC for Camarilla levels (using previous day's data)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels R4 and S4 (outer bands)
    r4 = pp + (prev_high - prev_low) * 1.5000
    s4 = pp - (prev_high - prev_low) * 1.5000
    r1 = pp + (prev_high - prev_low) * 1.0833
    s1 = pp - (prev_high - prev_low) * 1.0833
    
    # Align all indicators to 12h timeframe
    vwap_prev_aligned = align_htf_to_ltf(prices, df_1d, vwap_prev.values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_prev_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R4, price > 1d VWAP, volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > vwap_prev_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S4, price < 1d VWAP, volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < vwap_prev_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to central pivot or reverses to S1
            if (close[i] <= pp_aligned[i]) or (close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to central pivot or reverses to R1
            if (close[i] >= pp_aligned[i]) or (close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals