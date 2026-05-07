#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from daily timeframe with volume confirmation and 1d EMA trend filter.
# Enters long when price breaks above R3 level with volume above average and price > 1d EMA34.
# Enters short when price breaks below S3 level with volume above average and price < 1d EMA34.
# Uses tight entry conditions to limit trades to 20-40 per year, reducing fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's OHLC for Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day's high
    prev_low[0] = low_1d[0]    # First day uses same day's low
    prev_close[0] = close_1d[0] # First day uses same day's close
    
    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume confirmation
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: price breaks above R3 + price > 1d EMA34 + volume confirmation
            if (close[i] > camarilla_r3_4h[i] and 
                close[i] > ema_34_1d_aligned[i] and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + price < 1d EMA34 + volume confirmation
            elif (close[i] < camarilla_s3_4h[i] and 
                  close[i] < ema_34_1d_aligned[i] and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or trend changes
            if (close[i] < camarilla_s3_4h[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or trend changes
            if (close[i] > camarilla_r3_4h[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals