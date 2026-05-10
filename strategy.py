#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: Camarilla pivot breakouts in the direction of the 4h trend during active market hours.
Uses 4h for trend direction and 1h for entry timing. Designed for 15-37 trades/year per symbol.
Works in bull markets by buying breakouts above R1 in uptrends, and in bear markets by selling
breakdowns below S1 in downtrends. Volume confirmation reduces false breakouts.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate Camarilla levels from previous day's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # 4h trend: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_4h_up = close_4h > ema34_4h
    trend_4h_down = close_4h < ema34_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # Volume filter: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(trend_4h_up_aligned[i]) or
            np.isnan(trend_4h_down_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above R1 in 4h uptrend with volume
            if (close[i] > R1[i] and trend_4h_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 in 4h downtrend with volume
            elif (close[i] < S1[i] and trend_4h_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to mean (prev close) or trend changes
            if (close[i] <= prev_close[i] or trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to mean or trend changes
            if (close[i] >= prev_close[i] or trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals