#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (R3, S3) using previous day's OHLC
    prev_close = np.roll(df_1d['close'], 1)
    prev_high = np.roll(df_1d['high'], 1)
    prev_low = np.roll(df_1d['low'], 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need 34 for EMA + 1 for roll
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        
        if position == 0:
            # Enter long: Break above R3 in uptrend (price > EMA34) with volume confirmation
            if close[i] > r3 and close[i] > ema_1d and volume[i] > np.median(volume[max(0, i-24):i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Break below S3 in downtrend (price < EMA34) with volume confirmation
            elif close[i] < s3 and close[i] < ema_1d and volume[i] > np.median(volume[max(0, i-24):i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below EMA34 (trend change) or reaches S3 (stop)
            if close[i] < ema_1d or close[i] < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above EMA34 (trend change) or reaches R3 (stop)
            if close[i] > ema_1d or close[i] > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals