#!/usr/bin/env python3
"""
1d_1w_PriceChannel_Breakout_Volume_Trend
Hypothesis: Trade weekly Donchian channel breakouts on daily timeframe with volume confirmation and weekly trend filter. Enter long when price breaks above weekly Donchian high (20-period) with volume > 1.5x 20-day average and weekly EMA34 trending up. Enter short when price breaks below weekly Donchian low with volume confirmation and weekly EMA34 trending down. Target 10-25 trades/year via weekly breakout rarity. Works in bull/bear by following weekly trend. Uses volume confirmation to avoid false breakouts.
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
    
    # Get weekly data for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(34) for trend filter
    close_weekly = df_weekly['close'].values
    ema_period = 34
    ema_weekly = np.full_like(close_weekly, np.nan)
    
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period - 1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 / (ema_period + 1)) + (ema_weekly[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_period = 20
    donchian_high = np.full_like(high_weekly, np.nan)
    donchian_low = np.full_like(low_weekly, np.nan)
    
    for i in range(donchian_period - 1, len(high_weekly)):
        donchian_high[i] = np.max(high_weekly[i - donchian_period + 1:i + 1])
        donchian_low[i] = np.min(low_weekly[i - donchian_period + 1:i + 1])
    
    # Align weekly Donchian channels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price > weekly Donchian high + volume + weekly EMA trending up
            if (close[i] > donchian_high_aligned[i] and vol_confirm and 
                i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and ema_weekly_aligned[i] > ema_weekly_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price < weekly Donchian low + volume + weekly EMA trending down
            elif (close[i] < donchian_low_aligned[i] and vol_confirm and 
                  i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and ema_weekly_aligned[i] < ema_weekly_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < weekly Donchian low or weekly EMA turns down
            if close[i] < donchian_low_aligned[i] or (i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and ema_weekly_aligned[i] < ema_weekly_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > weekly Donchian high or weekly EMA turns up
            if close[i] > donchian_high_aligned[i] or (i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and ema_weekly_aligned[i] > ema_weekly_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_PriceChannel_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0