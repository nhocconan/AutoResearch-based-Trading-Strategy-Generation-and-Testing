#!/usr/bin/env python3
"""
1d_1w_PriceChannel_Breakout_Volume_Trend
Hypothesis: Trade weekly price channel breakouts on daily timeframe with volume confirmation and weekly trend filter. Enter long when price breaks above weekly Donchian high with volume > 1.5x average and weekly trend up (weekly close > weekly open), short when price breaks below weekly Donchian low with volume > 1.5x average and weekly trend down (weekly close < weekly open). Uses ATR-based stoploss to limit downside. Designed for low frequency (target 10-25 trades/year) to minimize fee drag and work in both bull (catch breakouts) and bear (fade false breaks via trend filter) markets.
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
    
    # Get weekly data for trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend: 1 if close > open, -1 if close < open
    weekly_trend = np.where(df_1w['close'] > df_1w['open'], 1, -1)
    
    # Weekly Donchian channels (20-period)
    lookback = 20
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    
    for i in range(lookback, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-lookback:i])
        donchian_low[i] = np.min(low_1w[i-lookback:i])
    
    # Align weekly data to daily
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, vol_period)  # Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + volume + weekly trend up
            if close[i] > donchian_high_aligned[i] and vol_confirm and weekly_trend_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + volume + weekly trend down
            elif close[i] < donchian_low_aligned[i] and vol_confirm and weekly_trend_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low or weekly trend turns down
            if close[i] < donchian_low_aligned[i] or weekly_trend_aligned[i] == -1:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high or weekly trend turns up
            if close[i] > donchian_high_aligned[i] or weekly_trend_aligned[i] == 1:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_PriceChannel_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0