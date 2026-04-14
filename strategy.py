#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w EMA200 trend filter
# Long when price breaks above 4h Donchian upper band with volume > 1.5x 1d average volume and price above weekly EMA200
# Short when price breaks below 4h Donchian lower band with volume > 1.5x 1d average volume and price below weekly EMA200
# Exit when price crosses the 4h Donchian midline
# Weekly EMA200 acts as a strong trend filter to avoid counter-trend trades in bear markets
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag and increase win rate

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h, 1d, and 1w data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h Donchian channel (20-period lookback)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate weekly EMA200
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200  # for 200-period EMA and 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and price above weekly EMA200
            if (price > donchian_upper_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume confirmation (using 1d average)
                price > ema_200_1w_aligned[i]):                 # Price above weekly EMA200 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and price below weekly EMA200
            elif (price < donchian_lower_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume confirmation (using 1d average)
                  price < ema_200_1w_aligned[i]):                 # Price below weekly EMA200 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_1wEMA200_1dVolume"
timeframe = "4h"
leverage = 1.0