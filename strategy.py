#!/usr/bin/env python3
"""
12h_1w_1d_camarilla_volume_v4
Hypothesis: Camarilla pivot levels on 1d with volume spike and 1w trend filter.
- Long: Price touches S3 level on 12h + volume > 1.5x 20-period average + 1w close > EMA50
- Short: Price touches R3 level on 12h + volume > 1.5x 20-period average + 1w close < EMA50
- Exit: Price crosses H4/L4 levels or trend reversal
- Position sizing: 0.25 long, -0.25 short
- Designed for range-bound markets with trend filter to avoid counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_volume_v4"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's range)
    # S1 = C - (H-L)*1.08/2, S2 = C - (H-L)*1.16/2, S3 = C - (H-L)*1.26/2
    # R1 = C + (H-L)*1.08/2, R2 = C + (H-L)*1.16/2, R3 = C + (H-L)*1.26/2
    # H4 = C + (H-L)*1.16/2, L4 = C - (H-L)*1.16/2
    rng = high_1d - low_1d
    s3 = close_1d - rng * 1.26 / 2
    r3 = close_1d + rng * 1.26 / 2
    h4 = close_1d + rng * 1.16 / 2
    l4 = close_1d - rng * 1.16 / 2
    
    # Align Camarilla levels to 12h
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    h4_12h = align_htf_to_ltf(prices, df_1d, h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema_50_1w
    trend_1w_down = close_1w < ema_50_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(s3_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below L4 OR 1w trend turns down
            if (close[i] < l4_12h[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price crosses above H4 OR 1w trend turns up
            if (close[i] > h4_12h[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price touches S3 + volume + 1w uptrend
            if (low[i] <= s3_12h[i]) and volume_filter[i] and trend_1w_up_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches R3 + volume + 1w downtrend
            elif (high[i] >= r3_12h[i]) and volume_filter[i] and trend_1w_down_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals