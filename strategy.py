#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Strategy: Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above R1 and above 1d EMA(34) with volume spike
# Short when price breaks below S1 and below 1d EMA(34) with volume spike
# Exit when price returns to H4/L4 or opposite Camarilla level is breached
# Uses proven Camarilla structure with trend and volume filters for robustness

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate volume average (20-period) for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for previous period
        # Use previous bar's high/low/close for current bar's levels
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        # Camarilla levels
        R4 = pclose + ((phigh - plow) * 1.5000)
        R3 = pclose + ((phigh - plow) * 1.2500)
        R2 = pclose + ((phigh - plow) * 1.1666)
        R1 = pclose + ((phigh - plow) * 1.0833)
        S1 = pclose - ((phigh - plow) * 1.0833)
        S2 = pclose - ((phigh - plow) * 1.1666)
        S3 = pclose - ((phigh - plow) * 1.2500)
        S4 = pclose - ((phigh - plow) * 1.5000)
        
        # Volume spike condition (2x average)
        volume_spike = volume[i] > (2.0 * vol_avg[i])
        
        if position == 0:
            # Enter long: price breaks above R1 with trend and volume
            if close[i] > R1 and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with trend and volume
            elif close[i] < S1 and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to H4 or breaks S1 (contrarian signal)
            if close[i] < R1 or close[i] < S1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to L4 or breaks R1 (contrarian signal)
            if close[i] > S1 or close[i] > R1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals