#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume average.
- Donchian Channel: identifies breakouts from 20-period price extremes.
- Entry: Long when price breaks above upper Donchian AND price > 1d EMA34 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below lower Donchian AND price < 1d EMA34 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture strong momentum moves that work in both trending and ranging markets.
- Volume confirmation ensures breakout legitimacy and reduces false signals.
- 1d EMA34 provides robust trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period=20):
    """Calculate Donchian Channel: returns upper, lower, middle."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Donchian Channels from 12h data (20-period)
    dc_upper, dc_lower, dc_middle = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Need 20 for Donchian, 34 for 1d EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(dc_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below lower Donchian
            if position == 1:
                if curr_close < dc_lower[i] and prev_close >= dc_lower[i-1]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper Donchian
            elif position == -1:
                if curr_close > dc_upper[i] and prev_close <= dc_upper[i-1]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_close > dc_upper[i] and prev_close <= dc_upper[i-1]
            breakout_down = curr_close < dc_lower[i] and prev_close >= dc_lower[i-1]
            
            # Trend filter: price vs 1d EMA34
            long_trend = curr_close > ema34_1d_aligned[i]
            short_trend = curr_close < ema34_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if breakout_up and long_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif breakout_down and short_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0