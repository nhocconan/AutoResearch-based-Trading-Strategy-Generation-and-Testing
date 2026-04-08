#!/usr/bin/env python3
"""
4h_12h_donchian_breakout_volume_v1
Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
- Long: Price breaks above Donchian(20) high + price > 12h EMA(50) + volume > 1.5x 20-period average
- Short: Price breaks below Donchian(20) low + price < 12h EMA(50) + volume > 1.5x 20-period average
- Exit: Opposite Donchian break or trend reversal
- Position sizing: 0.25 long, -0.25 short
- Designed to capture trends with volume confirmation, works in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    donchian_window = 20
    high_max = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_min = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema_50_12h
    trend_12h_down = close_12h < ema_50_12h
    
    # Forward fill trend
    trend_12h_up_series = pd.Series(trend_12h_up)
    trend_12h_down_series = pd.Series(trend_12h_down)
    trend_12h_up_ffilled = trend_12h_up_series.ffill().values
    trend_12h_down_ffilled = trend_12h_down_series.ffill().values
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up_ffilled)
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down_ffilled)
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = donchian_window
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR 12h trend turns down
            if close[i] <= low_min[i] or trend_12h_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR 12h trend turns up
            if close[i] >= high_max[i] or trend_12h_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high + volume + 12h uptrend
            if (close[i] >= high_max[i] and volume_filter[i] and trend_12h_up_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + volume + 12h downtrend
            elif (close[i] <= low_min[i] and volume_filter[i] and trend_12h_down_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals