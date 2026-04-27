#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with weekly volume confirmation and 1-week trend filter.
Enters long when price breaks above Donchian(20) upper band with volume above weekly average and 1-week uptrend.
Enters short when price breaks below Donchian(20) lower band with volume above weekly average and 1-week downtrend.
Trend filter uses 1-week EMA(34) to filter out countertrend moves.
Exit when price crosses the opposite Donchian band or trend reverses.
Target: 10-25 trades/year per symbol to minimize fee drift and work in both bull and bear markets.
"""

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
    
    # Get weekly data for volume filter and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly volume MA(20)
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) bands on daily data
    # Need 20 periods for Donchian calculation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after Donchian warmup (20 periods)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and weekly data
        price_now = close[i]
        vol_ma = vol_ma_20_1w_aligned[i]
        trend_1w = ema_34_1w_aligned[i]
        
        # Volume filter: volume > 1.5x weekly average
        vol_filter = prices['volume'].iloc[i] > 1.5 * vol_ma
        
        # Donchian breakout conditions with volume + trend filter
        if position == 0:
            # Long: price breaks above upper band with volume + weekly uptrend
            if price_now > highest_20[i] and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume + weekly downtrend
            elif price_now < lowest_20[i] and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower band or weekly trend turns down
            if price_now < lowest_20[i] or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper band or weekly trend turns up
            if price_now > highest_20[i] or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wVolume_1wTrend"
timeframe = "1d"
leverage = 1.0