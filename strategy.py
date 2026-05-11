#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_Volume_Confirmation_Trend_Filter
Hypothesis: Breakouts above 4h Donchian(20) high/low with volume > 2x 20-period average and trend filter (4h EMA50) capture institutional moves. Works in both bull (breakouts) and bear (breakdowns) markets by trading breakouts in direction of higher timeframe trend (12h EMA50). Uses 4h timeframe for signal generation with 12h trend filter to reduce whipsaw and focus on high-probability breaks. Designed for low trade frequency (<50/year) to minimize fee drag.
"""

name = "4h_Donchian_Breakout_20_Volume_Confirmation_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h trend filter (EMA 50)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h trend filter (EMA 50) - higher timeframe trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above Donchian high + above EMA50 + above 12h EMA50 + volume spike
            if (close[i] > high_roll[i] and 
                close[i] > ema_50[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + below EMA50 + below 12h EMA50 + volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_50[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to Donchian middle or trend reversal
            if position == 1:
                # Exit long: price returns to Donchian middle OR trend turns down
                donchian_mid = (high_roll[i] + low_roll[i]) / 2.0
                if (close[i] <= donchian_mid) or \
                   (close[i] < ema_50[i]) or \
                   (close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to Donchian middle OR trend turns up
                donchian_mid = (high_roll[i] + low_roll[i]) / 2.0
                if (close[i] >= donchian_mid) or \
                   (close[i] > ema_50[i]) or \
                   (close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals