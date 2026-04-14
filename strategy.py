#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter + volume confirmation
# Long when price breaks above Donchian high in uptrend, short when breaks below Donchian low in downtrend
# Trend: 1w EMA50 slope (rising/falling)
# Volume: > 1.5x 20-period average
# Designed for 100-150 trades over 4 years with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA50 on weekly
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Slope of EMA50: rising if current > previous
    ema50_slope = np.diff(ema50_1w, prepend=ema50_1w[0]) > 0
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1w EMA50 slope
        ema50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope)[i]
        
        # Check for NaN values
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_slope_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Uptrend: EMA50 rising -> look for long on breakout above Donchian high
                if ema50_slope_aligned:
                    if close[i] > donch_high[i]:
                        position = 1
                        signals[i] = position_size
                # Downtrend: EMA50 falling -> look for short on breakdown below Donchian low
                else:
                    if close[i] < donch_low[i]:
                        position = -1
                        signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below Donchian low
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above Donchian high
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0