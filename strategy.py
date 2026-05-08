#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and 4h volume confirmation.
# Long when price breaks above Donchian(20) upper band AND 1d EMA > 1d EMA previous (upward) AND 4h volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) lower band AND 1d EMA < 1d EMA previous (downward) AND 4h volume > 1.5x 20-period average.
# Exit when price crosses back below Donchian middle (for long) or above Donchian middle (for short).
# Uses Donchian breakouts for trend capture with EMA trend filter to avoid ranging markets.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drift.

name = "4h_Donchian_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA (34-period) on 1d data
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]  # First value
    
    # EMA trend: upward if current EMA > previous EMA
    ema_trend_up = ema_34 > ema_34_prev
    ema_trend_down = ema_34 < ema_34_prev
    
    # Align 1d EMA trend to 4h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_up)
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_trend_up_aligned[i]) or 
            np.isnan(ema_trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band, EMA trending up, volume spike
            long_cond = (close[i] > donchian_upper[i]) and ema_trend_up_aligned[i] and volume_filter[i]
            # Short conditions: break below lower band, EMA trending down, volume spike
            short_cond = (close[i] < donchian_lower[i]) and ema_trend_down_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below middle band
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above middle band
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals