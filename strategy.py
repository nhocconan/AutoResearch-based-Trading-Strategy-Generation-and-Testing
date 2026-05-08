#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian upper band (20) AND 1w EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20) AND 1w EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (between upper and lower bands).
# This strategy captures breakouts with higher timeframe trend alignment and volume confirmation.
# Daily timeframe reduces trade frequency to avoid fee drag while capturing major trends.
# Weekly EMA filter ensures we trade with the dominant trend, reducing whipsaws in choppy markets.
# Volume spike confirms institutional participation. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian_20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from daily data
    high_daily = high
    low_daily = low
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, 1w EMA50 rising, volume filter
            long_cond = (close[i] > donchian_high[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian low, 1w EMA50 falling, volume filter
            short_cond = (close[i] < donchian_low[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals