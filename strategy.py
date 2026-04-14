#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA trend filter and volume confirmation
# Uses price channel breakouts as primary signal with weekly trend filter to avoid counter-trend trades
# Volume confirmation reduces false breakouts. Works in bull/bear by only trading in direction of weekly trend.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on daily data
    # Upper channel: highest high of last 20 days
    # Lower channel: lowest low of last 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily Donchian levels to 1d timeframe (no additional delay needed for breakout)
    # The breakout uses the Donchian levels from the previous day to avoid look-ahead
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Align weekly EMA to 1d timeframe (no additional delay for trend filter)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20, 50)  # for 20-period Donchian and 20-period volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above weekly EMA50 with volume filter
            if (price > high_20_aligned[i] and price > ema_50_1w_aligned[i] and 
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian AND below weekly EMA50 with volume filter
            elif (price < low_20_aligned[i] and price < ema_50_1w_aligned[i] and 
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR below weekly EMA50
            if price < low_20_aligned[i] or price < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR above weekly EMA50
            if price > high_20_aligned[i] or price > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_Breakout_Weekly_EMA_Volume"
timeframe = "1d"
leverage = 1.0