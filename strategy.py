#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w EMA50 trend filter + 12h Donchian(20) breakout + volume confirmation.
Long when price breaks above 12h Donchian(20) high with 1w EMA50 uptrend and volume > 1.5x 20-period volume average.
Short when price breaks below 12h Donchian(20) low with 1w EMA50 downtrend and volume > 1.5x 20-period volume average.
Uses weekly EMA50 for primary trend direction to avoid counter-trend trades, reducing whipsaw in ranging markets.
Designed for low trade frequency (target: 12-37/year) with high conviction breakouts in BTC/ETH/SOL.
"""

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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 12h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian(20) high with 1w EMA50 uptrend and volume
            if (close[i] > donchian_upper[i] and 
                close_1w[i] > ema_50_1w[i] and  # 1w close above its EMA50 (uptrend)
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian(20) low with 1w EMA50 downtrend and volume
            elif (close[i] < donchian_lower[i] and 
                  close_1w[i] < ema_50_1w[i] and  # 1w close below its EMA50 (downtrend)
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 12h Donchian(20) low (opposite side of channel)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 12h Donchian(20) high (opposite side of channel)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wEMA50_TrendFilter_Donchian20_Breakout_Volume_Confirm"
timeframe = "12h"
leverage = 1.0