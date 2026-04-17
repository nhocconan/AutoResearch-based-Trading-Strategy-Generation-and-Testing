#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel AND volume > 1.8x 20-period average AND price > 1d EMA50 (bullish trend).
Short when price breaks below lower Donchian channel AND volume > 1.8x 20-period average AND price < 1d EMA50 (bearish trend).
Exit when price crosses the 1d EMA50 in opposite direction.
Donchian channels provide clear breakout levels, 1d EMA50 filters for primary trend direction,
volume confirmation reduces false breakouts. Designed for low trade frequency (20-50/year) to minimize fee drag.
Works in both bull markets (trend-following breakouts) and bear markets (shorting breakdowns with trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume > 1.8x avg AND price > 1d EMA50 (bullish trend)
            if high_price > upper and vol > 1.8 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND volume > 1.8x avg AND price < 1d EMA50 (bearish trend)
            elif low_price < lower and vol > 1.8 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA50
            if price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA50
            if price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_Filter"
timeframe = "4h"
leverage = 1.0