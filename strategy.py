#!/usr/bin/env python3
"""
12h_Weekly_PriceChannelBreakout_VolumeFilter
Hypothesis: Price breaking above/below weekly Donchian(20) channels on 12h timeframe with volume confirmation and weekly trend filter (price above/below weekly EMA200) creates high-probability trend-following signals. Weekly trend filter ensures alignment with long-term momentum, reducing false signals during corrections. Volume breakout confirms institutional participation. Target: 15-30 trades/year to minimize fee drag. Designed to work in both bull (breakouts up) and bear (breakdowns down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data once for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    weekly_close = df_weekly['close'].values
    ema200_weekly = np.zeros_like(weekly_close)
    ema200_weekly[0] = weekly_close[0]
    alpha = 2.0 / (200 + 1)
    for i in range(1, len(weekly_close)):
        ema200_weekly[i] = alpha * weekly_close[i] + (1 - alpha) * ema200_weekly[i-1]
    
    # Weekly Donchian channels (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    donchian_high = np.full_like(weekly_high, np.nan)
    donchian_low = np.full_like(weekly_low, np.nan)
    
    for i in range(len(weekly_high)):
        if i >= 19:  # 20 periods including current
            donchian_high[i] = np.max(weekly_high[i-19:i+1])
            donchian_low[i] = np.min(weekly_low[i-19:i+1])
        else:
            donchian_high[i] = np.max(weekly_high[:i+1])
            donchian_low[i] = np.min(weekly_low[:i+1])
    
    # Align weekly indicators to 12h timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period average (strict for low frequency)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (2.0 * volume_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after weekly EMA warmup
        # Skip if NaN in critical values
        if np.isnan(ema200_weekly_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema200 = ema200_weekly_aligned[i]
        upper_channel = donchian_high_aligned[i]
        lower_channel = donchian_low_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 3.0 * ATR from entry
        if position == 1 and price < entry_price - 3.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 3.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and weekly uptrend (price > weekly EMA200)
            if price > upper_channel and vol_ok and price > ema200:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly Donchian low with volume and weekly downtrend (price < weekly EMA200)
            elif price < lower_channel and vol_ok and price < ema200:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below weekly Donchian low (channel breakdown) or weekly trend turns down
            if price < lower_channel or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above weekly Donchian high (channel breakout) or weekly trend turns up
            if price > upper_channel or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Weekly_PriceChannelBreakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0