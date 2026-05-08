#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot (R3/S3) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 with 1d uptrend and volume > 1.5x average.
# Short when price breaks below S3 with 1d downtrend and volume > 1.5x average.
# Uses Camarilla levels from prior 1d to avoid look-ahead.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drag.
# Works in bull (breakouts continue) and bear (breakouts fail/reverse).

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on prior day)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, 1d uptrend, volume confirmation
            if (close[i] > r3_aligned[i] and 
                close[i-1] <= r3_aligned[i-1] and  # Ensure breakout just happened
                close[i] > ema_34_aligned[i] and   # Price above 1d EMA (uptrend)
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, 1d downtrend, volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  close[i-1] >= s3_aligned[i-1] and  # Ensure breakdown just happened
                  close[i] < ema_34_aligned[i] and   # Price below 1d EMA (downtrend)
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (failed breakout) or trend reversal
            if (close[i] < s3_aligned[i] and 
                close[i-1] >= s3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_aligned[i]:  # Trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 (failed breakdown) or trend reversal
            if (close[i] > r3_aligned[i] and 
                close[i-1] <= r3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_aligned[i]:  # Trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals