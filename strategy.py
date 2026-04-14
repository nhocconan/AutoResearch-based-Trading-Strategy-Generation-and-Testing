#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with weekly trend filter (1w EMA50) and volume confirmation
# Long when price breaks above 12h Donchian upper band with volume >1.5x 20-period average and price above weekly EMA50
# Short when price breaks below 12h Donchian lower band with volume >1.5x 20-period average and price below weekly EMA50
# Exit when price crosses the 12h Donchian midline
# Weekly EMA50 acts as a trend filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and weekly data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate 12h Donchian channel (20-period lookback)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume[i]  # Current 12h volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and price above weekly EMA50
            if (price > donchian_upper_aligned[i] and 
                vol_12h_current > 1.5 * vol_ma_12h_aligned[i] and  # Volume confirmation
                price > ema50_aligned[i]):                      # Price above weekly EMA50 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and price below weekly EMA50
            elif (price < donchian_lower_aligned[i] and 
                  vol_12h_current > 1.5 * vol_ma_12h_aligned[i] and  # Volume confirmation
                  price < ema50_aligned[i]):                      # Price below weekly EMA50 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0