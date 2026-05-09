#!/usr/bin/env python3
# 1h_4hDonchianBreakout_1dTrend_20Vol
# Hypothesis: 4h Donchian breakout (20-period) with 1d EMA200 trend filter and 20-period volume spike.
# 1h used only for entry timing. Works in bull markets by buying breakouts in uptrends, in bear markets by selling breakdowns in downtrends.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 15-35 trades/year on 1h timeframe.

name = "1h_4hDonchianBreakout_1dTrend_20Vol"
timeframe = "1h"
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
    
    # Get 4h data for Donchian breakout levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period high/low)
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(low_4h, np.nan)
    
    if len(high_4h) >= 20:
        for i in range(len(high_4h)):
            if i >= 19:
                donchian_high[i] = np.max(high_4h[i-19:i+1])
                donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(200) with proper initialization
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 + ema_200_1d[i-1] * 198) / 200
    
    # Align 1d EMA to 1h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: 1h volume / 20-period average volume
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
           np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above 4h Donchian high AND volume confirmation AND bullish trend (price > EMA200)
            if close[i] > donchian_high_aligned[i] and volume_ratio[i] > 2.0 and close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: Price breaks below 4h Donchian low AND volume confirmation AND bearish trend (price < EMA200)
            elif close[i] < donchian_low_aligned[i] and volume_ratio[i] > 2.0 and close[i] < ema_200_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below 4h Donchian low (reversal signal) or trend turns bearish
            if close[i] < donchian_low_aligned[i] or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price breaks above 4h Donchian high (reversal signal) or trend turns bullish
            if close[i] > donchian_high_aligned[i] or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals