#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum breakout with 4h trend filter and daily volume confirmation
# In trending markets (4h EMA), breakouts above/below 1h Donchian channels with volume surge capture momentum.
# Works in bull/bear by using 4h EMA trend filter and requiring volume confirmation > 2x average.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h EMA(50) for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily average volume (30-period) for confirmation
    volume_1d_series = pd.Series(volume_1d)
    avg_vol_1d = volume_1d_series.rolling(window=30, min_periods=30).mean().shift(1).values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 1h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(50, 30, 20)  # for 50-period EMA, 30-period volume, 20-period Donchian
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above 1h Donchian high AND above 4h EMA50 with volume surge
            if (price > donchian_high[i] and price > ema_50_4h_aligned[i] and 
                vol > 2.0 * avg_vol_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1h Donchian low AND below 4h EMA50 with volume surge
            elif (price < donchian_low[i] and price < ema_50_4h_aligned[i] and 
                  vol > 2.0 * avg_vol_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 1h Donchian low OR below 4h EMA50
            if price < donchian_low[i] or price < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 1h Donchian high OR above 4h EMA50
            if price > donchian_high[i] or price > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Donchian_Breakout_4hEMA_Volume"
timeframe = "1h"
leverage = 1.0