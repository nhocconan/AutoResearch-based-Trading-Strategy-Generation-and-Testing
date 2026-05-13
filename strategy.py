#!/usr/bin/env python3
"""
4H_Donchian_Volume_Trend_Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume surge confirms institutional participation, and EMA(50) trend filter ensures we only trade in the direction of the higher timeframe trend. Works in bull markets by catching breakouts and in bear markets by catching breakdowns with volume confirmation.
"""

name = "4H_Donchian_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian Channel (20-period)
    donchian_period = 20
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    
    for i in range(donchian_period - 1, len(high)):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Calculate EMA(50) for trend filter
    ema_period = 50
    ema = np.full_like(close, np.nan)
    if len(close) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    # Get 1d data for volume average and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period volume average on daily
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate EMA(50) on daily for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_50_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] - ema_50_1d[i-1]) * multiplier + ema_50_1d[i-1]
    
    # Align 1d indicators to 4h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period daily average
        vol_spike = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + price above EMA(50) + 1d EMA uptrend
            if (close[i] > upper_channel[i] and vol_spike and 
                close[i] > ema[i] and close_1d[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume spike + price below EMA(50) + 1d EMA downtrend
            elif (close[i] < lower_channel[i] and vol_spike and 
                  close[i] < ema[i] and close_1d[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or loss of volume/momentum
            if (close[i] < lower_channel[i] or not vol_spike or close[i] < ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or loss of volume/momentum
            if (close[i] > upper_channel[i] or not vol_spike or close[i] > ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals