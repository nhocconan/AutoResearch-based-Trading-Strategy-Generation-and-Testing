#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band, price above weekly EMA20 (bullish trend), and volume >1.5x 20-day average
# Short when price breaks below 1d Donchian lower band, price below weekly EMA20 (bearish trend), and volume >1.5x 20-day average
# Exit when price crosses the 1d Donchian midline
# Weekly EMA20 acts as trend filter to avoid counter-trend trades
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channel (20-period lookback)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    donchian_upper = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    weekly_ema20 = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-day average volume for confirmation
    vol_daily = df_daily['volume'].values
    vol_ma_20 = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_daily, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_daily, donchian_middle)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]  # Current daily volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and bullish weekly trend
            if (price > donchian_upper_aligned[i] and 
                vol_current > 1.5 * vol_ma_20_aligned[i] and  # Volume confirmation
                price > weekly_ema20_aligned[i]):             # Price above weekly EMA20 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and bearish weekly trend
            elif (price < donchian_lower_aligned[i] and 
                  vol_current > 1.5 * vol_ma_20_aligned[i] and  # Volume confirmation
                  price < weekly_ema20_aligned[i]):             # Price below weekly EMA20 for bearish bias
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

name = "1d_Donchian_WeeklyEMA20_Volume"
timeframe = "1d"
leverage = 1.0