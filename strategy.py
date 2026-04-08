#!/usr/bin/env python3
"""
12h_1d_keltner_breakout_volume_v1
Hypothesis: Trend continuation on 12h with Keltner channel breakout confirmed by volume and 1d EMA trend.
- Entry: Price breaks above/below Keltner(20,2.0) + volume > 1.5x 20-period average + 1d EMA(50) trend aligned
- Exit: Price returns to Keltner middle (EMA20) or 1d trend reverses
- Position sizing: 0.30 long/short, 0.0 flat
- Target: 20-40 trades/year (80-160 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema_50_1d
    trend_1d_down = close_1d < ema_50_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # 12h Keltner Channel (20, 2.0)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    middle_keltner = ema_20
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below middle Keltner OR 1d trend turns down
            if close[i] < middle_keltner[i] or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price above middle Keltner OR 1d trend turns up
            if close[i] > middle_keltner[i] or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Position size
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper Keltner + 1d uptrend + volume
            if close[i] > upper_keltner[i] and trend_1d_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.30
            # Short entry: Price breaks below lower Keltner + 1d downtrend + volume
            elif close[i] < lower_keltner[i] and trend_1d_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.30
    
    return signals