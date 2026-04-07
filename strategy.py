#!/usr/bin/env python3
"""
12h_donchian_20_1w_trend_volume_v1
Hypothesis: On 12-hour timeframe, use 20-period Donchian channel breakouts with 1-week trend filter and volume confirmation.
Long when price breaks above upper Donchian band with weekly EMA(50) trending up and volume > 1.5x 20-period average.
Short when price breaks below lower Donchian band with weekly EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price returns to the middle of the Donchian channel (20-period SMA of high/low average).
Designed for 15-30 trades/year to minimize fee drag while capturing strong trends with multi-timeframe confirmation.
Works in both bull/bear markets as Donchian channels adapt to volatility and weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    for i in range(1, len(ema_50_1w_aligned)):
        if not np.isnan(ema_50_1w_aligned[i]) and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_trend_down[i] = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
    
    # Calculate Donchian Channel on 12h timeframe (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Middle line for exit (average of high and low)
    donchian_middle = (highest_high + lowest_low) / 2
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to middle of Donchian channel
            if close[i] <= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle of Donchian channel
            if close[i] >= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: price breaks above upper Donchian band with weekly uptrend
                if (close[i] > highest_high[i] and close[i-1] <= highest_high[i-1] and 
                    weekly_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band with weekly downtrend
                elif (close[i] < lowest_low[i] and close[i-1] >= lowest_low[i-1] and 
                      weekly_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals