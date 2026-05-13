#!/usr/bin/env python3
name = "6H_ElderRay_1D_Trend_Volume"
timeframe = "6h"
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
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = np.zeros_like(close)
    ema13[:] = np.nan
    alpha = 2 / (13 + 1)
    for i in range(len(close)):
        if i == 0:
            ema13[i] = close[i]
        elif np.isnan(ema13[i-1]):
            ema13[i] = close[i]
        else:
            ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA34 on daily close
    ema_34 = np.zeros_like(close_1d)
    ema_34[:] = np.nan
    alpha_34 = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_34[i] = close_1d[i]
        elif np.isnan(ema_34[i-1]):
            ema_34[i] = close_1d[i]
        else:
            ema_34[i] = alpha_34 * close_1d[i] + (1 - alpha_34) * ema_34[i-1]
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate volume average (20-period) for volume confirmation
    vol_ma_20 = np.zeros_like(volume)
    vol_ma_20[:] = np.nan
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Bull Power > 0 + Bear Power < 0 (Elder Ray bullish) + 1d uptrend + volume
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_34_aligned[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
            # SHORT: Bull Power < 0 + Bear Power > 0 (Elder Ray bearish) + 1d downtrend + volume
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < ema_34_aligned[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power becomes positive (bullish momentum fading)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power becomes negative (bearish momentum fading)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals