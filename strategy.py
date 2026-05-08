#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    R1 = prev_close + 1.1 * (high_1d - low_1d) / 12
    S1 = prev_close - 1.1 * (high_1d - low_1d) / 12
    R2 = prev_close + 1.1 * (high_1d - low_1d) / 6
    S2 = prev_close - 1.1 * (high_1d - low_1d) / 6
    
    # Align Camarilla levels to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(R2_4h[i]) or 
            np.isnan(S2_4h[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above daily EMA34 + volume confirmation
            if (close[i] > R1_4h[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond R2
                if close[i] <= R2_4h[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S1 + below daily EMA34 + volume confirmation
            elif (close[i] < S1_4h[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond S2
                if close[i] >= S2_4h[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below S1 OR below daily EMA34
            if close[i] < S1_4h[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above R1 OR above daily EMA34
            if close[i] > R1_4h[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals