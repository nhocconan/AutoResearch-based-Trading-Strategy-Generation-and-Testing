#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# In trending markets, price breaks above/below Donchian channels with momentum.
# In ranging markets, breakouts fail quickly. The 1w EMA filter ensures we only
# trade in the direction of the higher timeframe trend, reducing false breakouts.
# Volume confirmation adds validity to breakouts. Target: 15-25 trades per year (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(30) for 1w trend filter
    ema30_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (30 + 1)
    ema30_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema30_1w[i] = (close_1w[i] - ema30_1w[i-1]) * ema_multiplier + ema30_1w[i-1]
    
    # Align 1w EMA to 1d timeframe
    ema30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema30_1w)
    
    # Donchian channels (20-period) on 1d timeframe
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema30_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema30_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper + above 1w EMA + volume confirmation
            if (price > highest_high[i] and
                price > ema_trend and
                vol > 1.5 * avg_vol):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian lower + below 1w EMA + volume confirmation
            elif (price < lowest_low[i] and
                  price < ema_trend and
                  vol > 1.5 * avg_vol):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below Donchian lower or closes below 1w EMA
            if (price < lowest_low[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above Donchian upper or closes above 1w EMA
            if (price > highest_high[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Trend_Volume"
timeframe = "1d"
leverage = 1.0