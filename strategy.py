#146644
# 1d_Donchian20_Breakout_1wTrend_Volume
# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation.
# Uses weekly trend to capture long-term momentum, Donchian(20) breakouts for entry timing,
# and volume confirmation to filter false breakouts. Designed to generate ~10-25 trades/year
# on 1d timeframe to avoid fee drag while maintaining edge in both bull and bear markets.
# Long when weekly trend up (weekly close > weekly EMA20), price breaks above Donchian(20) high,
# and volume > 2x average. Short when weekly trend down (weekly close < weekly EMA20),
# price breaks below Donchian(20) low, and volume > 2x average.

name = "1d_Donchian20_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema20_1w[i] = (close_1w[i] * 2 + ema20_1w[i-1] * 18) / 20
    
    # Align weekly EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(n):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume vs 20-day average
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
    
    start_idx = max(19, 19)  # Need Donchian(20) and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema20_1w_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above Donchian high + volume confirmation
            if trend_up and close[i] > donchian_high[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below Donchian low + volume confirmation
            elif not trend_up and close[i] < donchian_low[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below Donchian low
            if not trend_up or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above Donchian high
            if trend_up or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals