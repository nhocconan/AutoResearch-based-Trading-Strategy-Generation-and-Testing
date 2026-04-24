#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter and volume average.
- Donchian Channel: identifies breakouts from 20-day high/low.
- Entry: Long when price breaks above upper Donchian AND price > 1w EMA50 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below lower Donchian AND price < 1w EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Donchian breakout signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakout works in both bull and bear markets by capturing sustained momentum.
- Volume confirmation ensures breakout legitimacy.
- 1w EMA50 provides robust trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period=20):
    """Calculate Donchian Channels: returns upper, lower."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1w volume average for confirmation
    if len(df_1w) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    # Calculate Donchian Channels from 1d data (20-period)
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Need 20 for Donchian, 50 for 1w EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below lower Donchian
            if position == 1:
                if curr_close < dc_lower[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper Donchian
            elif position == -1:
                if curr_close > dc_upper[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_close > dc_upper[i] and prev_close <= dc_upper[i-1]
            breakout_down = curr_close < dc_lower[i] and prev_close >= dc_lower[i-1]
            
            # Trend filter: price vs 1w EMA50
            long_trend = curr_close > ema50_1w_aligned[i]
            short_trend = curr_close < ema50_1w_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
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

name = "1d_Donchian20_Breakout_1wEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0