#!/usr/bin/env python3
# 12h_WeeklyDonchian_Breakout_1wTrend_VolumeFilter
# Hypothesis: Breakout above/below weekly Donchian(20) levels with volume >1.5x 20-bar average and trend filter from 1w EMA20.
# Works in bull markets by buying breakouts in uptrends, in bear markets by selling breakdowns in downtrends.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 15-30 trades/year on 12h timeframe.

name = "12h_WeeklyDonchian_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian breakout levels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian(20) - highest high and lowest low over past 20 weekly bars
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 20:
        for i in range(20, len(close_1w)):
            donchian_high[i] = np.max(high_1w[i-20:i])
            donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Align 1w Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 1w EMA(20) for trend filter
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20
    
    # Align 1w EMA to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above weekly Donchian high AND volume confirmation AND bullish trend (price > EMA)
            if close[i] > donchian_high_aligned[i] and volume_ratio[i] > 1.5 and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below weekly Donchian low AND volume confirmation AND bearish trend (price < EMA)
            elif close[i] < donchian_low_aligned[i] and volume_ratio[i] > 1.5 and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below weekly Donchian low (reversal signal) or trend turns bearish
            if close[i] < donchian_low_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above weekly Donchian high (reversal signal) or trend turns bullish
            if close[i] > donchian_high_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals