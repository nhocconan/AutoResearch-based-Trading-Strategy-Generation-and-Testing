#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Camarilla pivot levels from previous 1d
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_r4 = np.zeros(len(close_1d))
    camarilla_s4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_r4[i] = np.nan
            camarilla_s4[i] = np.nan
            continue
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_r3[i] = close_1d[i-1] + range_ * 1.1000 / 4
        camarilla_s3[i] = close_1d[i-1] - range_ * 1.1000 / 4
        camarilla_r4[i] = close_1d[i-1] + range_ * 1.1000 / 2
        camarilla_s4[i] = close_1d[i-1] - range_ * 1.1000 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike detection: volume > 1.5 * average volume over last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 AND volume spike AND 1d uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND volume spike AND 1d downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 OR opposite signal with volume
            if (close[i] < camarilla_s3_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 OR opposite signal with volume
            if (close[i] > camarilla_r3_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals