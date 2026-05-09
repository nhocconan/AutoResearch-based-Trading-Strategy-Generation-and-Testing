#!/usr/bin/env python3
# 1d_WeeklyDonchian20_Breakout_1wTrend_VolumeSpike
# Hypothesis: Weekly Donchian breakout with weekly trend filter and daily volume confirmation.
# Uses weekly trend to capture long-term direction, weekly breakout for entry timing,
# and daily volume spike to filter for institutional participation. Works in bull/bear:
# trend filter prevents counter-trend trades, volume spike confirms breakout strength.
# Weekly timeframe reduces trade frequency to avoid fee drag while capturing major moves.

name = "1d_WeeklyDonchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
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
    
    # Calculate weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period rolling high/low for weekly Donchian
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    
    if len(high_1w) >= 20:
        for i in range(19, len(high_1w)):
            donchian_high[i] = np.max(high_1w[i-19:i+1])
            donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate weekly EMA for trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_21_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 21:
        ema_21_1w[20] = np.mean(close_1w[0:21])
        for i in range(21, len(close_1w)):
            ema_21_1w[i] = (ema_21_1w[i-1] * 20 + close_1w[i]) / 21
    
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure volume MA and weekly indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high AND uptrend (price > weekly EMA) AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_21_1w_aligned[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low AND downtrend (price < weekly EMA) AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_21_1w_aligned[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low OR trend reversal (price < weekly EMA)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high OR trend reversal (price > weekly EMA)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals