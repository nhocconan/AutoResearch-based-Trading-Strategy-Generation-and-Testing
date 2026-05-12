#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
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
    
    # 1d trend filter: EMA34 (Higher timeframe trend)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d Camarilla levels (R3, S3)
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    camarilla_high = df_1d_close + (df_1d_high - df_1d_low) * 1.1 / 4
    camarilla_low = df_1d_close - (df_1d_high - df_1d_low) * 1.1 / 4
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume filter: 4h volume > 1.5 * 20-period SMA
    volume_series = pd.Series(volume)
    vol_sma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for 1d EMA34 and volume SMA
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R3 + 1d uptrend + volume spike
            if (close[i] > camarilla_high_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > 1.5 * vol_sma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 + 1d downtrend + volume spike
            elif (close[i] < camarilla_low_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > 1.5 * vol_sma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price closes below S3 or 1d trend turns down
            if (close[i] < camarilla_low_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price closes above R3 or 1d trend turns up
            if (close[i] > camarilla_high_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals