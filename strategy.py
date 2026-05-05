#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period high) AND price > EMA50(12h) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band (20-period low) AND price < EMA50(12h) AND volume > 1.5x 20-period average
# Exit when price crosses back below/above Donchian middle (10-period average) OR trend flips (price crosses EMA50(12h))
# Donchian channels provide clear structure for breakouts in both bull and bear markets
# 12h EMA50 avoids counter-trend whipsaws during ranging periods
# Volume confirmation ensures institutional participation
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) for 4h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 4h data (20-period)
    if len(high) >= 20 and len(low) >= 20:
        # Upper band: 20-period high
        upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower band: 20-period low
        lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Middle band: 10-period average (for exit)
        middle_band = (pd.Series(high).rolling(window=10, min_periods=10).mean().values + 
                      pd.Series(low).rolling(window=10, min_periods=10).mean().values) / 2
    else:
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        middle_band = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND price > EMA50(12h) AND volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND price < EMA50(12h) AND volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian middle band (mean reversion) OR price < EMA50(12h) (trend flip)
            if (close[i] < middle_band[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian middle band (mean reversion) OR price > EMA50(12h) (trend flip)
            if (close[i] > middle_band[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals