#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_Volume_Confirmation_Trend_Filter
Hypothesis: Price breaking above 4-hour Donchian high (20-period) or below Donchian low with volume confirmation and 1-day EMA trend filter. Works in bull markets via breakouts and in bear markets via breakdowns. Designed for low frequency (20-50 trades/year) with high win rate by requiring volume spike and trend alignment.
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
    
    # 4h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1-day trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: break above Donchian high + above 1d EMA50 + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + below 1d EMA50 + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to opposite Donchian band or trend reversal
            if position == 1:
                # Exit long: price returns to Donchian low OR trend turns down
                if (close[i] <= donchian_low[i]) or \
                   (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to Donchian high OR trend turns up
                if (close[i] >= donchian_high[i]) or \
                   (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals