#!/usr/bin/env python3
"""
4h_1d_1w_PriceChannel_Breakout_Volume_Regime_v1
Hypothesis: Breakout of 1d high/low with 1w trend filter and volume confirmation.
Works in bull by following 1w trend upward breakouts, works in bear by taking 1w trend downward breakdowns.
Volume filters out false breakouts. Target: 20-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for price channel (high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high/low for channel
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Align to 4h
    channel_high = align_htf_to_ltf(prices, df_1d, prev_high)
    channel_low = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Load 1w data for trend (close > SMA50 = uptrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50 = pd.Series(close_1w).rolling(50, min_periods=50).mean().values
    trend_up = close_1w > sma_50  # True for uptrend
    
    # Align trend to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(channel_high[i]) or np.isnan(channel_low[i]) or 
            np.isnan(trend_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: breakout above 1d high in uptrend with volume
            if (price > channel_high[i] and trend_up_aligned[i] and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below 1d low in downtrend with volume
            elif (price < channel_low[i] and not trend_up_aligned[i] and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below 1d high (failed breakout) or reverse signal
            if price < channel_high[i] or (not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above 1d low (failed breakdown) or reverse signal
            if price > channel_low[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_PriceChannel_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0