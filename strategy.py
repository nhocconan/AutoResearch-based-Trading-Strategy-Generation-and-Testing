# 4h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike
# Uses Camarilla pivot levels from daily data for precise entry/exit levels.
# Long when price breaks above R3 with 1d EMA34 uptrend and volume > 2x average.
# Short when price breaks below S3 with 1d EMA34 downtrend and volume > 2x average.
# Exit when price returns to the central pivot (P) level.
# Targets 20-50 trades/year to minimize fee drag while maintaining edge.

#!/usr/bin/env python3
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
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    camarilla_R3 = np.full(len(close_1d), np.nan)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    camarilla_P = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            # First day has no previous data
            continue
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_hl = H - L
        
        camarilla_P[i] = (H + L + C) / 3
        camarilla_R3[i] = C + (range_hl * 1.2500)
        camarilla_S3[i] = C - (range_hl * 1.2500)
    
    # Align 1d indicators to 4h timeframe
    camarilla_R3_4h = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_4h = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_P_4h = align_htf_to_ltf(prices, df_1d, camarilla_P)
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup period
    start_idx = max(19, 1)  # Need volume MA20 and previous day's pivot
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3_4h[i]) or np.isnan(camarilla_S3_4h[i]) or 
            np.isnan(camarilla_P_4h[i]) or np.isnan(ema_1d_4h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume spike (at least 2x average)
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above R3 with 1d EMA34 uptrend and volume spike
            if (price > camarilla_R3_4h[i] and 
                price > ema_1d_4h[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with 1d EMA34 downtrend and volume spike
            elif (price < camarilla_S3_4h[i] and 
                  price < ema_1d_4h[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to or below central pivot (P)
            if price <= camarilla_P_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to or above central pivot (P)
            if price >= camarilla_P_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0