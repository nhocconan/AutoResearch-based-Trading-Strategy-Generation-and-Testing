#!/usr/bin/env python3
name = "6h_Liquidity_Zone_Breakout_1wTrend_OrderFlow"
timeframe = "6h"
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
    
    # Get weekly data for liquidity zones and trend
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly liquidity zones: previous week's high/low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Align liquidity zones to 6h timeframe (previous week's values)
    liquidity_high = align_htf_to_ltf(prices, df_1w, weekly_high)
    liquidity_low = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Weekly trend filter: EMA50 of weekly close
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Order flow proxy: volume-weighted price change
    # Positive when price rises on high volume, negative when falls on high volume
    price_change = np.diff(close, prepend=close[0])
    volume_weighted_change = price_change * volume
    # Smooth to get order flow direction
    order_flow = pd.Series(volume_weighted_change).ewm(span=10, min_periods=10).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(liquidity_high[i]) or np.isnan(liquidity_low[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(order_flow[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly liquidity high AND uptrend AND positive order flow AND volume surge
            if (close[i] > liquidity_high[i] and 
                close[i] > weekly_ema50_aligned[i] and 
                order_flow[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly liquidity low AND downtrend AND negative order flow AND volume surge
            elif (close[i] < liquidity_low[i] and 
                  close[i] < weekly_ema50_aligned[i] and 
                  order_flow[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly liquidity low OR trend turns down
            if (close[i] < liquidity_low[i] or 
                close[i] < weekly_ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly liquidity high OR trend turns up
            if (close[i] > liquidity_high[i] or 
                close[i] > weekly_ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals