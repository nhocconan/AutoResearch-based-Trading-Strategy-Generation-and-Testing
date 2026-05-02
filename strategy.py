#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and weekly pivot direction
# Uses weekly pivot levels from 1d data (previous week's close) for directional bias:
#   - Long only if price > weekly pivot (bullish bias)
#   - Short only if price < weekly pivot (bearish bias)
# Weekly pivot acts as a regime filter to avoid counter-trend trades in strong moves
# Donchian(20) breakout provides entry timing with volume confirmation (2.0x 24-period average)
# Designed for low trade frequency (~12-25/year) to minimize fee drag on 6h timeframe
# Works in bull markets via trend-aligned breakouts above weekly pivot
# Works in bear markets via short breakdowns below weekly pivot with Donchian low breaks

name = "6h_Donchian20_1wPivot_Direction_Volume_Confirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA trend filter and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot from 1d data (previous week's close)
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    # We approximate weekly close as the close 5 days ago (5 * 1d bars)
    weekly_close = np.roll(close_1d, 5)
    # For weekly high/low, we use rolling max/min over 5 periods
    weekly_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian H20 + price > 1d EMA34 + price > weekly pivot + volume confirm
            if close[i] > donchian_high[i] and close[i] > ema_34_1d_aligned[i] and close[i] > weekly_pivot_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian L20 + price < 1d EMA34 + price < weekly pivot + volume confirm
            elif close[i] < donchian_low[i] and close[i] < ema_34_1d_aligned[i] and close[i] < weekly_pivot_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian L20 or weekly pivot
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian H20 or weekly pivot
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals