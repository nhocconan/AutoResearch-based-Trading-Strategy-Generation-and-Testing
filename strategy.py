#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week EMA50 trend filter + Donchian(20) breakout + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses Donchian(20) midpoint OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w EMA50 filters for primary weekly trend alignment to avoid counter-trend trades
# Donchian breakout provides clear structure-based entries in both bull and bear markets
# Volume confirmation ensures breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels on 1d
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high, above 1w EMA50, volume confirmation, in session
            if (close[i] > highest_high_20[i] and close[i] > ema50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low, below 1w EMA50, volume confirmation, in session
            elif (close[i] < lowest_low_20[i] and close[i] < ema50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals