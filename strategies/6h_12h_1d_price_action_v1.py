#!/usr/bin/env python3
"""
6h_12h_1d_price_action_v1
Hypothesis: Use 12h and 1d timeframes to establish trend and key levels, then trade breakouts on 6h.
- Trend: 12h close above/below 12h EMA(50) determines long/short bias
- Support/Resistance: 1d high/low act as dynamic levels
- Entry: 6h price breaks 1d high/low in direction of 12h trend with volume confirmation
- Exit: Opposite 1d level touch or trend reversal
- Volume: Require 6h volume > 1.5x 20-period average to avoid false breakouts
Target: 12-37 trades/year (50-150 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_price_action_v1"
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
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_12h > ema_50
    trend_down = close_12h < ema_50
    
    # Forward fill trend
    trend_up_series = pd.Series(trend_up)
    trend_down_series = pd.Series(trend_down)
    trend_up_ffilled = trend_up_series.ffill().values
    trend_down_ffilled = trend_down_series.ffill().values
    
    # Align 12h trend to 6h
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up_ffilled)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down_ffilled)
    
    # Get 1d data for support/resistance
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily high/low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Forward fill daily levels
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    daily_high_ffilled = daily_high_series.ffill().values
    daily_low_ffilled = daily_low_series.ffill().values
    
    # Align daily levels to 6h
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high_ffilled)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low_ffilled)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches daily low (support) or trend turns down
            if low[i] <= daily_low_aligned[i] or trend_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price touches daily high (resistance) or trend turns up
            if high[i] >= daily_high_aligned[i] or trend_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above daily high with 12h uptrend and volume
            if high[i] > daily_high_aligned[i] and trend_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below daily low with 12h downtrend and volume
            elif low[i] < daily_low_aligned[i] and trend_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals