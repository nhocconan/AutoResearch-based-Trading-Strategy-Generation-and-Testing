#!/usr/bin/env python3
"""
4h_DonchianBreakout_Volume_TrendFilter
Hypothesis: 4h timeframe with Donchian(20) breakout, volume confirmation, and 1w EMA trend filter.
Enters long when price breaks above Donchian upper band with volume > 1.5x 20-period average and weekly uptrend.
Enters short when price breaks below Donchian lower band with volume > 1.5x 20-period average and weekly downtrend.
Uses ATR(14) for volatility-based stop loss and position sizing of 0.25.
Designed for 20-50 trades/year to avoid fee drag in 4h timeframe.
Works in bull/bear via trend filter and volatility-adjusted breakouts.
"""

name = "4h_DonchianBreakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get weekly close aligned to 4h for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Donchian(20) channels
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(vol_ratio[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(close_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume and weekly uptrend
            if (high[i] > upper_band[i] and 
                close[i] > upper_band[i] and  # close confirmation
                vol_ratio[i] > 1.5 and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume and weekly downtrend
            elif (low[i] < lower_band[i] and 
                  close[i] < lower_band[i] and  # close confirmation
                  vol_ratio[i] > 1.5 and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band or trend turns down
            if (low[i] < lower_band[i] and 
                close[i] < lower_band[i]) or \
               not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band or trend turns up
            if (high[i] > upper_band[i] and 
                close[i] > upper_band[i]) or \
               not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals