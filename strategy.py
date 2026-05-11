#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume_Trend
Hypothesis: Daily trend filter + 4h Donchian breakout with volume confirmation.
Uses 1-day EMA34 for trend direction, 4-hour Donchian(20) breakout for entry,
and volume spike (2x 24-period average) for confirmation. Exits on opposite Donchian breakout or trend reversal.
Designed to work in both bull and bear markets by following daily trend, avoiding counter-trend trades.
Targets low trade frequency (20-40/year) via daily trend filter and breakout logic.
"""

name = "4h_1d_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily EMA34 for Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = np.where(close_1d > ema_34_1d, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Align daily trend to 4h timeframe
    trend_1d_4h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # --- 4h Donchian(20) for Entry/Exit ---
    period = 20
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Volume Spike Detection (24-period average = 4 days on 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_1d_4h[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        # Daily trend direction
        daily_trend = trend_1d_4h[i]
        
        if position == 0:
            # Long: daily uptrend + price breaks above upper Donchian + volume
            if (daily_trend == 1 and 
                close[i] > upper_channel[i] and 
                close[i-1] <= upper_channel[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + price breaks below lower Donchian + volume
            elif (daily_trend == -1 and 
                  close[i] < lower_channel[i] and 
                  close[i-1] >= lower_channel[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: daily trend turns down OR price breaks below lower Donchian
                if (trend_1d_4h[i] == -1 or 
                    close[i] < lower_channel[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: daily trend turns up OR price breaks above upper Donchian
                if (trend_1d_4h[i] == 1 or 
                    close[i] > upper_channel[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals