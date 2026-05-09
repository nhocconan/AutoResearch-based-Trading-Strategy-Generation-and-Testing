#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter
# Hypothesis: Use 4h EMA20 for trend direction and 1d volume spike for confirmation, with 1h entries at Camarilla R1/S1 breakouts.
# Works in bull markets by buying breakouts in uptrends, in bear markets by selling breakdowns in downtrends.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 15-30 trades/year on 1h timeframe.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter"
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
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(20) with proper initialization
    ema_20_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        ema_20_4h[19] = np.mean(close_4h[0:20])
        for i in range(20, len(close_4h)):
            ema_20_4h[i] = (close_4h[i] * 2 + ema_20_4h[i-1] * 18) / 20
    
    # Align 4h EMA to 1h timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume EMA(20)
    vol_ema_20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ema_20_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ema_20_1d[i] = (volume_1d[i] * 2 + vol_ema_20_1d[i-1] * 18) / 20
    
    # Align 1d volume EMA to 1h timeframe
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Calculate Camarilla levels from previous day's OHLC (using 1d data)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev = np.roll(high_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev = np.roll(low_1d, 1)
    low_1d_prev[0] = np.nan
    
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    valid = ~np.isnan(close_1d_prev) & ~np.isnan(high_1d_prev) & ~np.isnan(low_1d_prev)
    camarilla_R1[valid] = close_1d_prev[valid] + (high_1d_prev[valid] - low_1d_prev[valid]) * 1.1 / 12
    camarilla_S1[valid] = close_1d_prev[valid] - (high_1d_prev[valid] - low_1d_prev[valid]) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume ratio: current 1h volume / 20-period average 1h volume
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma_20[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma_20)) & (vol_ma_20 != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma_20[valid_vol]
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]) or \
           not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above R1 AND volume confirmation AND bullish trend (price > 4h EMA)
            if close[i] > camarilla_R1_aligned[i] and volume_ratio[i] > 2.0 and close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: Price breaks below S1 AND volume confirmation AND bearish trend (price < 4h EMA)
            elif close[i] < camarilla_S1_aligned[i] and volume_ratio[i] > 2.0 and close[i] < ema_20_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price breaks above R1 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals