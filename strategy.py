#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hEMA200_Trend_Volume"
timeframe = "1h"
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
    
    # 1d data for Camarilla pivot calculation (uses previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for each day
    # Using previous day's data for today's levels (avoid look-ahead)
    # R1 = C + ((H-L) * 1.1/6)
    # S1 = C - ((H-L) * 1.1/6)
    # where C, H, L are from previous day
    
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    
    # 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)  # Moderate volume spike
    
    # Align all indicators to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(2, 200, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and above 4h EMA200
            if (close[i] > camarilla_r1_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema_200_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume spike and below 4h EMA200
            elif (close[i] < camarilla_s1_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema_200_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price closes below R1 or 4h EMA200
            if close[i] < camarilla_r1_aligned[i] or close[i] < ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price closes above S1 or 4h EMA200
            if close[i] > camarilla_s1_aligned[i] or close[i] > ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals