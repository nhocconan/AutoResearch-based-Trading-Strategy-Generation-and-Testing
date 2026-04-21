# 1. Hypothesis:
# Strategy type: 12h timeframe with 1h/4h multi-timeframe confluence, using Williams %R for mean reversion and Donchian breakout for trend confirmation.
# Why it should work in both bull and bear: Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets, while Donchian breakout with volume confirmation captures strong trends. The combination adapts to market regimes, reducing whipsaws in sideways markets and capturing momentum in trending markets. Volume confirmation filters low-conviction moves. Designed for moderate trade frequency (~15-30 trades/year) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1h data ONCE before loop for entry timing
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Williams %R(14) on 1h for mean reversion signals
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_1h_aligned = align_htf_to_ltf(prices, df_1h, williams_r)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Donchian Channel(20) on 4h for trend confirmation
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over past 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over past 20 periods
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Volume confirmation: current volume vs 20-period average on 1h
    vol_1h = df_1h['volume'].values
    vol_ma_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1h / vol_ma_20
    vol_ratio_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_1h_aligned[i]) or 
            np.isnan(donchian_upper_4h_aligned[i]) or 
            np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(vol_ratio_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        williams_r_val = williams_r_1h_aligned[i]
        upper_band = donchian_upper_4h_aligned[i]
        lower_band = donchian_lower_4h_aligned[i]
        vol_ratio_val = vol_ratio_1h_aligned[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + price above Donchian lower band + volume confirmation
            if (williams_r_val < -80 and 
                price_close > lower_band and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + price below Donchian upper band + volume confirmation
            elif (williams_r_val > -20 and 
                  price_close < upper_band and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: Williams %R overbought (> -20) or price breaks below Donchian lower band
                if williams_r_val > -20 or price_close < lower_band:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                # Exit short: Williams %R oversold (< -80) or price breaks above Donchian upper band
                if williams_r_val < -80 or price_close > upper_band:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals

name = "12h_WilliamsR_Donchian_Volume_Confluence"
timeframe = "12h"
leverage = 1.0