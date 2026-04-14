#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly trend filter and volume confirmation
# Long when price breaks above 1d 20-day Donchian high AND weekly EMA50 is rising AND volume > 1.5x 20-day average
# Short when price breaks below 1d 20-day Donchian low AND weekly EMA50 is falling AND volume > 1.5x 20-day average
# Exit when price returns to the 10-day EMA (mean reversion to short-term trend)
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while capturing trends in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d 10-period EMA for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).values
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).values
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1d timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_10_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: break above Donchian high, weekly uptrend, high volume
            if (price > high_20_aligned[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and  # weekly EMA rising
                volume[i] > 1.5 * vol_ma_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low, weekly downtrend, high volume
            elif (price < low_20_aligned[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and  # weekly EMA falling
                  volume[i] > 1.5 * vol_ma_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 10-day EMA
            if price <= ema_10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 10-day EMA
            if price >= ema_10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0