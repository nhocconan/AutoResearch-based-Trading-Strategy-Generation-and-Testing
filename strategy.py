#!/usr/bin/env python3
# 1H_Camarilla_R3S3_4hTrend_VolumeFilter
# Hypothesis: 1-hour Camarilla R3/S3 breakout with 4-hour trend filter (price above/below 4h EMA34) and volume spike confirmation.
# Uses 4h trend to avoid counter-trend trades in both bull and bear markets.
# Volume spike ensures momentum confirmation. Targets 15-35 trades/year to minimize fee drag.
# Uses discrete position sizing (0.20).

name = "1H_Camarilla_R3S3_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels on 1h data (using previous period's high-low-close)
    # Camarilla: H = high, L = low, C = close
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    lookback = 1
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        H = high[i-1]
        L = low[i-1]
        C = close[i-1]
        camarilla_r3[i] = C + (H - L) * 1.1 / 2
        camarilla_s3[i] = C - (H - L) * 1.1 / 2
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure we have EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.5x average volume)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 + 4h uptrend + volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_4h_aligned[i] and   # 4h uptrend filter
                volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 + 4h downtrend + volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_4h_aligned[i] and   # 4h downtrend filter
                  volume_filter):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: Price returns to the middle of Camarilla range (H-L)/2 + C
            camarilla_mid = (camarilla_r3[i] + camarilla_s3[i]) / 2
            range_width = camarilla_r3[i] - camarilla_s3[i]
            at_mid = abs(close[i] - camarilla_mid) < range_width * 0.25  # Within 25% of range
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals