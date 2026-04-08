#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 4h timeframe, price breaking above/below Donchian(20) channel with volume expansion and daily trend alignment captures breakout moves. Daily trend filter avoids counter-trend breakouts in ranging markets. Volume confirmation filters false breakouts.
- Long: Price > Donchian upper(20) + volume > 1.5x 20-period average + daily uptrend
- Short: Price < Donchian lower(20) + volume > 1.5x 20-period average + daily downtrend
- Exit: Opposite Donchian level touch or daily trend reversal
- Position sizing: 0.25 long, -0.25 short
- Designed for trending markets while avoiding false breakouts in ranges
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
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
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(20) for trend
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1d_up = close_1d > ema_20_1d
    trend_1d_down = close_1d < ema_20_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches Donchian lower OR daily trend turns down
            if (close[i] <= low_roll[i]) or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price touches Donchian upper OR daily trend turns up
            if (close[i] >= high_roll[i]) or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price > Donchian upper + volume + daily uptrend
            if (close[i] > high_roll[i]) and volume_filter[i] and trend_1d_up_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price < Donchian lower + volume + daily downtrend
            elif (close[i] < low_roll[i]) and volume_filter[i] and trend_1d_down_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals