#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12-period Donchian channels on 12h (20 periods for breakout)
    donchian_period = 20
    dc_high = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    dc_low = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema_50_12h
    trend_12h_down = close_12h < ema_50_12h
    
    # Forward fill trend
    trend_12h_up_series = pd.Series(trend_12h_up)
    trend_12h_down_series = pd.Series(trend_12h_down)
    trend_12h_up_ffilled = trend_12h_up_series.ffill().values
    trend_12h_down_ffilled = trend_12h_down_series.ffill().values
    
    # Align 12h indicators to 6h
    dc_high_aligned = align_htf_to_ltf(prices, df_12h, dc_high)
    dc_low_aligned = align_htf_to_ltf(prices, df_12h, dc_low)
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up_ffilled)
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down_ffilled)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(dc_high_aligned[i]) or np.isnan(dc_low_aligned[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns down
            if close[i] <= dc_low_aligned[i] or trend_12h_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns up
            if close[i] >= dc_high_aligned[i] or trend_12h_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + 12h uptrend + volume
            if close[i] >= dc_high_aligned[i] and trend_12h_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + 12h downtrend + volume
            elif close[i] <= dc_low_aligned[i] and trend_12h_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals