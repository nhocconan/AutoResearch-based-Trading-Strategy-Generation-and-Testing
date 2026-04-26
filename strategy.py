#!/usr/bin/env python3
"""
6h_WeeklyDonchian20_Breakout_1dTrendFilter_VolumeSpike_v2
Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 6h timeframe targeting 75-150 total trades over 4 years (19-38/year)
- Long when price breaks above 20-period high with volume spike and 1d uptrend (close > EMA34)
- Short when price breaks below 20-period low with volume spike and 1d downtrend (close < EMA34)
- Weekly trend filter (1w EMA50) added to avoid counter-trend trades in strong weekly trends
- Volume spike confirms institutional participation (1.5x 20-period average)
- Designed for low trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Load 1w data ONCE before loop for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for weekly trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (1.5x 20-period volume average on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 34 for daily EMA, 20 for Donchian/volume)
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and trend filters
        price_above_high = close[i] > highest_20[i]
        price_below_low = close[i] < lowest_20[i]
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Weekly trend filter (avoid counter-trend trades)
        weekly_trend_up = close[i] > ema50_1w_aligned[i]
        weekly_trend_down = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-period high AND volume spike AND 1d uptrend AND weekly uptrend
            if price_above_high and volume_spike[i] and trend_up and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low AND volume spike AND 1d downtrend AND weekly downtrend
            elif price_below_low and volume_spike[i] and trend_down and weekly_trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below 20-period low OR 1d trend turns down
            if price_below_low or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above 20-period high OR 1d trend turns up
            if price_above_high or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dTrendFilter_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0