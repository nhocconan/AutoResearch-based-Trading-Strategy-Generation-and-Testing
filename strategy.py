#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily volume surge and 1-hour trend alignment.
# Uses 1-hour EMA50 for trend filter and daily volume > 2x 20-day average for confirmation.
# Long when price breaks above 4h Donchian upper (20) + 1h uptrend + volume surge.
# Short when price breaks below 4h Donchian lower (20) + 1h downtrend + volume surge.
# Exits on trend reversal or price retracement to 4h midline.
# Designed for low trade frequency (<50/year) to minimize fee drag and work in both bull/bear markets.

name = "4h_Donchian20_1hEMA50_VolumeSurge"
timeframe = "4h"
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
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Breakout conditions
    breakout_up = close > donchian_high
    breakout_down = close < donchian_low
    
    # Get 1h data for EMA50 trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 1h EMA50 trend filter
    ema_50_1h = pd.Series(df_1h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    trend_up = close > ema_50_1h_aligned
    trend_down = close < ema_50_1h_aligned
    
    # Volume filter: current volume > 2.0x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + 1h uptrend + volume surge
            if breakout_up[i] and trend_up[i] and volume_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + 1h downtrend + volume surge
            elif breakout_down[i] and trend_down[i] and volume_surge[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midline or trend reversal
            if close[i] <= donchian_mid[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement to midline or trend reversal
            if close[i] >= donchian_mid[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals