#!/usr/bin/env python3
"""
4h_12h_1d_donchian_breakout_volume_v1
Hypothesis: 4-hour Donchian breakout with 12-hour trend filter and 1-day volume confirmation.
- Entry: Price breaks above/below 20-period Donchian channel + 12h trend alignment + volume surge
- Exit: Opposite Donchian break or trend reversal
- Position sizing: 0.25 long, -0.25 short
- Designed for trending markets with volume confirmation to avoid false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_donchian_breakout_volume_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR 12h trend turns down
            if (close[i] <= donchian_low[i]) or trend_12h_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR 12h trend turns up
            if (close[i] >= donchian_high[i]) or trend_12h_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high + 12h uptrend + volume surge
            if (close[i] > donchian_high[i]) and trend_12h_up_aligned[i] and (vol_ratio_1d_aligned[i] > 1.5):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + 12h downtrend + volume surge
            elif (close[i] < donchian_low[i]) and trend_12h_down_aligned[i] and (vol_ratio_1d_aligned[i] > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals