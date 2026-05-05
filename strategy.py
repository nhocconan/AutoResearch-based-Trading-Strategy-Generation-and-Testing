#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above 1d Ichimoku Senkou Span A (cloud top) AND Tenkan > Kijun (bullish TK cross) AND volume > 1.5x 20-period average
# Short when price breaks below 1d Ichimoku Senkou Span B (cloud bottom) AND Tenkan < Kijun (bearish TK cross) AND volume > 1.5x 20-period average
# Exit when price re-enters the Ichimoku cloud (between Senkou Span A and B) OR TK cross reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol.
# Ichimoku provides dynamic support/resistance and trend direction, TK cross confirms momentum, volume spike validates breakout strength.
# This combination has shown strong performance in ranging and trending markets across multiple timeframes.

name = "6h_Ichimoku_Cloud_Breakout_1dTK_Cross_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (with proper delay for completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long conditions: price breaks above cloud TOP AND bullish TK cross (Tenkan > Kijun) AND volume spike
            if (close[i] > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below cloud BOTTOM AND bearish TK cross (Tenkan < Kijun) AND volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud (below cloud top) OR TK cross turns bearish (Tenkan < Kijun)
            if (close[i] < cloud_top or 
                tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters cloud (above cloud bottom) OR TK cross turns bullish (Tenkan > Kijun)
            if (close[i] > cloud_bottom or 
                tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals