#!/usr/bin/env python3
"""
6h Weekly Donchian Breakout with Daily Volume Spike and 1d EMA34 Trend Filter
Hypothesis: Weekly Donchian channels (20-period) capture major structural breaks.
In 6h timeframe, we trade breakouts of the weekly channel only when:
1. Price breaks above weekly Donchian high (long) or below weekly Donchian low (short)
2. Confirmed by 1d volume spike (>2.0x 20-period average)
3. Filtered by 1d EMA34 trend (price > EMA34 for longs, < EMA34 for shorts)
This avoids false breakouts in ranging markets and targets sustained moves.
Weekly HTF ensures we only trade with the larger trend, reducing whipsaw.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
Works in both bull (breakout continuation) and bear (breakdown continuation) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend and volume MA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA on 1d for volume confirmation
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get weekly data for Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high_20_1w = np.full(len(df_1w), np.nan)
    donchian_low_20_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        donchian_high_20_1w[i] = np.max(high_1w[i-19:i+1])
        donchian_low_20_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20_1w)
    donchian_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 20)  # 34 for EMA, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(donchian_high_20_1w_aligned[i]) or np.isnan(donchian_low_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        donchian_high = donchian_high_20_1w_aligned[i]
        donchian_low = donchian_low_20_1w_aligned[i]
        
        # Volume confirmation: current 6h volume > 2.0 * daily 20-period average
        # Note: Comparing 6h volume to daily average volume - this works as a relative spike filter
        volume_confirm = curr_volume > 2.0 * vol_ma_1d
        
        if position == 0:
            # Look for entry signals
            # Long: Price breaks above weekly Donchian high AND volume confirmation AND price > daily EMA34 (uptrend)
            long_entry = (curr_close > donchian_high and volume_confirm and curr_close > ema_trend)
            # Short: Price breaks below weekly Donchian low AND volume confirmation AND price < daily EMA34 (downtrend)
            short_entry = (curr_close < donchian_low and volume_confirm and curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price breaks below weekly Donchian low (contrarian exit) OR daily EMA34 turns down
            if curr_close < donchian_low or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price breaks above weekly Donchian high (contrarian exit) OR daily EMA34 turns up
            if curr_close > donchian_high or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0