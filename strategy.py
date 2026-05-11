#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below 20-period Donchian channel on 12h timeframe with 1d trend confirmation (EMA 34) and volume spike (>2x average). Uses tight entry conditions to limit trades to 15-35 per year. Trend filter ensures we only trade in direction of higher timeframe momentum. Volume confirms institutional interest. Designed for low frequency with high win rate in trending markets, avoiding false breakouts in chop.
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 12h Donchian channel (20-period)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 1d trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (24-period average on 12h = ~12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
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
            # Long: break above Donchian high + above 1d EMA34 + volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + below 1d EMA34 + volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to opposite Donchian band or trend reversal
            if position == 1:
                # Exit long: price returns to Donchian low OR trend turns down
                if (close[i] <= lowest_low[i]) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to Donchian high OR trend turns up
                if (close[i] >= highest_high[i]) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals