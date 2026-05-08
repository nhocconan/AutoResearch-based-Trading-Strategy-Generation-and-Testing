#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Channels with 1-day Volume-Weighted RSI Filter
# Long when price breaks above Donchian(20) upper band AND daily RSI(14) < 50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) lower band AND daily RSI(14) > 50 AND volume > 1.5x 20-bar avg
# Donchian provides breakout signals with defined risk; daily RSI filters against overextended moves
# Volume confirmation ensures institutional participation; avoids low-liquidity false breakouts
# Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

name = "4h_Donchian_DailyRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_1d_aligned[i]
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: price breaks above upper band AND daily RSI < 50 AND volume filter
            if price > upper and rsi_val < 50 and vol_filt:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band AND daily RSI > 50 AND volume filter
            elif price < lower and rsi_val > 50 and vol_filt:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below midpoint of channel
            midpoint = (upper + lower) / 2.0
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above midpoint of channel
            midpoint = (upper + lower) / 2.0
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals