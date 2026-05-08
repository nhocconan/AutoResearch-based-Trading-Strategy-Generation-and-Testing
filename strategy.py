#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot S1/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above S3 with 1d uptrend and volume spike.
# Short when price breaks below S1 with 1d downtrend and volume spike.
# Uses volume confirmation to avoid false breakouts.
# Target: 20-50 trades per year to minimize fee drag.
# Works in bull (breakout continuation) and bear (breakdown continuation) via trend filter.

name = "4h_Camarilla_S1S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: S1 = C - (H-L)*1.1/12, S3 = C - (H-L)*1.1/4
    #          R1 = C + (H-L)*1.1/12, R3 = C + (H-L)*1.1/4
    camarilla_s1 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_r1 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        hl = high_1d[i-1] - low_1d[i-1]
        camarilla_s1[i] = close_1d[i-1] - hl * 1.1 / 12
        camarilla_s3[i] = close_1d[i-1] - hl * 1.1 / 4
        camarilla_r1[i] = close_1d[i-1] + hl * 1.1 / 12
        camarilla_r3[i] = close_1d[i-1] + hl * 1.1 / 4
    
    # Align Camarilla levels to 4h (wait for 1d bar to close)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 4h (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i-1] <= camarilla_r3_aligned[i-1] and  # crossed above
                ema_34_aligned[i] > ema_34_aligned[i-1] and  # 1d EMA rising
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i-1] >= camarilla_s3_aligned[i-1] and  # crossed below
                  ema_34_aligned[i] < ema_34_aligned[i-1] and  # 1d EMA falling
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below R1 or trend reverses
            if (close[i] < camarilla_r1_aligned[i] and 
                close[i-1] >= camarilla_r1_aligned[i-1]) or ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above S1 or trend reverses
            if (close[i] > camarilla_s1_aligned[i] and 
                close[i-1] <= camarilla_s1_aligned[i-1]) or ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals