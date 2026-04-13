#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Donchian breakouts capture breakout moves; EMA filter ensures trend alignment;
# volume confirms breakout strength. Target: 30-100 total trades over 4 years (7-25/year)
# for 1d timeframe. Works in both bull (breakouts) and bear (mean reversion via exits).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(20) for 1w trend filter
    ema20_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    ema20_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * ema_multiplier + ema20_1w[i-1]
    
    # Align 1w EMA to 1d timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Donchian channels on 1d timeframe (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price breaks above upper Donchian + above 1w EMA + volume confirmation
            if (price > upper[i] and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian + below 1w EMA + volume confirmation
            elif (price < lower[i] and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below lower Donchian or below 1w EMA
            if (price < lower[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above upper Donchian or above 1w EMA
            if (price > upper[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Trend_Volume"
timeframe = "1d"
leverage = 1.0