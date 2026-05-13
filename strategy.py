#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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
    
    # Calculate Camarilla levels from previous day
    close_1d = np.zeros(n)
    high_1d = np.zeros(n)
    low_1d = np.zeros(n)
    for i in range(n):
        # Use daily data from previous close
        if i == 0:
            close_1d[i] = close[0]
            high_1d[i] = high[0]
            low_1d[i] = low[0]
        else:
            # Check if current bar is first 4h bar of a new day
            if prices['open_time'].iloc[i].date() != prices['open_time'].iloc[i-1].date():
                close_1d[i] = close[i-1]  # Previous bar's close as yesterday's close
                high_1d[i] = high[i-1]    # Previous bar's high as yesterday's high
                low_1d[i] = low[i-1]      # Previous bar's low as yesterday's low
            else:
                close_1d[i] = close_1d[i-1]
                high_1d[i] = high_1d[i-1]
                low_1d[i] = low_1d[i-1]
    
    # Calculate Camarilla R1, S1 levels
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # 1d trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d_vals = df_1d['close'].values
    ema34_1d = pd.Series(close_1d_vals).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R1 with 1d uptrend and volume spike
            if close[i] > R1[i] and close[i] > ema34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with 1d downtrend and volume spike
            elif close[i] < S1[i] and close[i] < ema34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1
            if close[i] < S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1
            if close[i] > R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals