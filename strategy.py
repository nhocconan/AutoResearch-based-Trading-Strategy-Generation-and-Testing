#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: Breakouts above Camarilla R3 or below S3 with 1d EMA34 trend filter and volume confirmation.
# Works in bull markets via R3 breakouts and bear markets via S3 breakdowns. Volume filter reduces false signals.
# Target: ~25-35 trades/year to stay within fee-efficient range.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate for each 1d bar: (H-L)/12
    range_1d = high_1d - low_1d
    camarilla_multiplier = range_1d / 12.0
    
    # Camarilla levels for current day based on previous day's action
    # R3 = C + (H-L)*1.1/12*4 = C + 1.1*(H-L)/3
    # S3 = C - (H-L)*1.1/12*4 = C - 1.1*(H-L)/3
    r3 = close_1d_prev + 1.1 * camarilla_multiplier * 4
    s3 = close_1d_prev - 1.1 * camarilla_multiplier * 4
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend and volume
            if (close[i] > r3_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume
            elif (close[i] < s3_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below S3 or trend reversal
            if (close[i] < s3_aligned[i] or
                trend_1d_down_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above R3 or trend reversal
            if (close[i] > r3_aligned[i] or
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals