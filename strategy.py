# WARNING: This is a placeholder for educational purposes only.
# Do NOT use this code for live trading.
# Backtest results are hypothetical and do not guarantee future performance.
# Trading involves substantial risk of loss.
# Past performance is not indicative of future results.
# Always conduct your own research and consult with a qualified financial advisor.

#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_With_Volume_and_Trend_Filter_v2
Hypothesis: Tighten the original strategy to reduce trade frequency while maintaining edge.
- Use 24-hour R1/S1 pivots instead of 12h (fewer signals)
- Add volume > 2.0x 50-period average (stricter volume filter)
- Require price to be outside Bollinger Bands(20,2) on 4h for entry (avoid chop)
- Keep 24h EMA50 trend filter
- Use discrete position sizes: 0.0, ±0.25
- Target: 150-250 total trades over 4 years (~38-63/year) to stay under 400 max
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 24-hour data for pivot and trend calculation (less frequent than 12h)
    df_24h = get_htf_data(prices, '24h')
    if len(df_24h) < 2:
        return np.zeros(n)
    
    # 24-hour OHLC arrays
    high_24h = df_24h['high'].values
    low_24h = df_24h['low'].values
    close_24h = df_24h['close'].values
    
    # Calculate Camarilla pivot levels for 24h timeframe
    pivot_24h = (high_24h + low_24h + close_24h) / 3
    r1_24h = close_24h + (high_24h - low_24h) * 1.1 / 12
    s1_24h = close_24h - (high_24h - low_24h) * 1.1 / 12
    
    # Align 24h levels to 4h timeframe
    r1_24h_aligned = align_htf_to_ltf(prices, df_24h, r1_24h)
    s1_24h_aligned = align_htf_to_ltf(prices, df_24h, s1_24h)
    pivot_24h_aligned = align_htf_to_ltf(prices, df_24h, pivot_24h)
    
    # 24-hour EMA trend filter (50-period)
    ema_24h = pd.Series(close_24h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_24h_aligned = align_htf_to_ltf(prices, df_24h, ema_24h)
    
    # 4h volume filter: >2.0x 50-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # 4h Bollinger Bands for chop avoidance (outside bands = trending)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_upper_arr = bb_middle.values + (2 * bb_std.values)
    bb_lower_arr = bb_middle.values - (2 * bb_std.values)
    outside_bands = (close > bb_upper_arr) | (close < bb_lower_arr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for volume MA and BB
    
    for i in range(start_idx, n):
        if (np.isnan(r1_24h_aligned[i]) or np.isnan(s1_24h_aligned[i]) or 
            np.isnan(ema_24h_aligned[i]) or np.isnan(volume_filter[i]) or
            np.isnan(outside_bands[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_24h_aligned[i]
        s1 = s1_24h_aligned[i]
        ema_trend = ema_24h_aligned[i]
        vol_ok = volume_filter[i]
        bb_ok = outside_bands[i]
        
        if position == 0:
            # Long: price breaks above 24h R1 with volume and outside BB
            if price > r1 and vol_ok and bb_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24h S1 with volume and outside BB
            elif price < s1 and vol_ok and bb_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below 24h pivot or trend reverses
            if price < pivot_24h_aligned[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above 24h pivot or trend reverses
            if price > pivot_24h_aligned[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_With_Volume_and_Trend_Filter_v2"
timeframe = "4h"
leverage = 1.0