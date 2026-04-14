#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and daily trend filter
# Long when price breaks above 4h Donchian upper band with volume >1.5x 20-period average and close > daily EMA200
# Short when price breaks below 4h Donchian lower band with volume >1.5x 20-period average and close < daily EMA200
# Exit when price crosses the 4h Donchian midline
# Daily EMA200 acts as a trend filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drain
# Works in bull (breakouts with trend) and bear (mean-reversion via exits at midline)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and daily data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channel (20-period lookback)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate daily EMA200 for trend filter
    close_daily = df_daily['close'].values
    ema200_daily = pd.Series(close_daily).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    ema200_aligned = align_htf_to_ltf(prices, df_daily, ema200_daily)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and above daily EMA200
            if (price > donchian_upper_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                price > ema200_aligned[i]):                     # Above daily EMA200 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and below daily EMA200
            elif (price < donchian_lower_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  price < ema200_aligned[i]):                     # Below daily EMA200 for bearish bias
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

name = "4h_Donchian_DailyEMA200_Volume"
timeframe = "4h"
leverage = 1.0