#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: Uses Camarilla pivot levels (R3/S3) from daily timeframe combined with 1d EMA34 trend filter and volume confirmation.
# Only enters on breakout of R3 (long) or S3 (short) when price is aligned with 1d EMA34 and volume is above average.
# Designed for 12h timeframe to target 12-37 trades/year, avoiding fee drag while capturing trend moves.
# Works in bull markets via breakouts and in bear markets via short breakdowns.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use previous day's data to avoid look-ahead
    camarilla_R3 = np.zeros(len(close_1d))
    camarilla_S3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC (i-1)
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_hl = H - L
        camarilla_R3[i] = C + range_hl * 1.1 / 4
        camarilla_S3[i] = C - range_hl * 1.1 / 4
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 12h timeframe
    camarilla_R3_12h = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_12h = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_R3_12h[i]) or np.isnan(camarilla_S3_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: breakout above R3 + price above EMA34 + volume confirmation
            if (close[i] > camarilla_R3_12h[i] and 
                close[i] > ema_34_12h[i] and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 + price below EMA34 + volume confirmation
            elif (close[i] < camarilla_S3_12h[i] and 
                  close[i] < ema_34_12h[i] and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or trend changes
            if (close[i] < camarilla_S3_12h[i] or 
                close[i] < ema_34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or trend changes
            if (close[i] > camarilla_R3_12h[i] or 
                close[i] > ema_34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals