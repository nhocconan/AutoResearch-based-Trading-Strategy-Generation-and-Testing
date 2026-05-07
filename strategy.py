#!/usr/bin/env python3
name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Donchian and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Donchian upper: highest high of last 20 days
    upper = np.full_like(d_high, np.nan)
    lower = np.full_like(d_low, np.nan)
    for i in range(20, len(d_high)):
        upper[i] = np.max(d_high[i-20:i])
        lower[i] = np.min(d_low[i-20:i])
    
    # Align to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_1d, upper)
    lower_6h = align_htf_to_ltf(prices, df_1d, lower)
    
    # Daily EMA34 for trend filter
    d_close = df_1d['close'].values
    ema_34_1d = np.full_like(d_close, np.nan)
    if len(d_close) >= 34:
        ema = np.zeros_like(d_close)
        ema[0] = d_close[0]
        for i in range(1, len(d_close)):
            ema[i] = (d_close[i] * 2/35) + (ema[i-1] * (1 - 2/35))
        ema_34_1d = ema
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_sum = np.zeros_like(volume)
        for i in range(len(volume)):
            if i < 20:
                vol_sum[i] = np.sum(volume[:i+1])
            else:
                vol_sum[i] = np.sum(volume[i-19:i+1])
        vol_ma_20 = vol_sum / np.minimum(np.arange(len(volume)) + 1, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above upper band in daily uptrend with volume
            if close[i] > upper_6h[i] and ema_34_6h[i] > ema_34_6h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band in daily downtrend with volume
            elif close[i] < lower_6h[i] and ema_34_6h[i] < ema_34_6h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to lower band or trend reverses
            if close[i] < lower_6h[i] or ema_34_6h[i] < ema_34_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to upper band or trend reverses
            if close[i] > upper_6h[i] or ema_34_6h[i] > ema_34_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian breakouts with trend filter and volume confirmation
# - Donchian(20) on 1d provides structure based on 20-day high/low
# - Breakout above upper band in daily uptrend (EMA34 rising) signals bullish continuation
# - Breakdown below lower band in daily downtrend (EMA34 falling) signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses 1d timeframe for structure and trend, 6h for execution timing
# - Novel combination: Donchian breakout with EMA trend filter and volume spike
# - Avoids saturated Donchian/volume families by using daily timeframe for structure