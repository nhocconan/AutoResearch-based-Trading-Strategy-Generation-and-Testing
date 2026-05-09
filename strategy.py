#!/usr/bin/env python3
# 1h_Camarilla_R3S3_Breakout_4hTrend_Volume
# Hypothesis: Breakout above/below daily Camarilla R3/S3 levels with volume >1.8x 20-bar average and trend filter from 4h EMA50.
# Camarilla levels provide high-probability reversal/breakout zones. In uptrend (price > EMA50), buy breakout above R3; in downtrend (price < EMA50), sell breakdown below S3.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 15-37 trades/year on 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA(50) with proper initialization
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (close_4h[i] * 2 + ema_50_4h[i-1] * 48) / 50
    
    # Align 4h EMA to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    daily_range = high_1d - low_1d
    camarilla_R3 = close_1d + daily_range * 1.1 / 2
    camarilla_S3 = close_1d - daily_range * 1.1 / 2
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume filter: 1h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or \
           np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Enter long: Price breaks above Camarilla R3 AND volume confirmation AND bullish trend (price > EMA50)
            if close[i] > camarilla_R3_aligned[i] and volume_ratio[i] > 1.8 and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: Price breaks below Camarilla S3 AND volume confirmation AND bearish trend (price < EMA50)
            elif close[i] < camarilla_S3_aligned[i] and volume_ratio[i] > 1.8 and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Camarilla S3 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price breaks above Camarilla R3 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals