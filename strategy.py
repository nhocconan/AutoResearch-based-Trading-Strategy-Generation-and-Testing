#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Long when: price > Ichimoku cloud (Senkou Span A & B), Tenkan > Kijun, price > 1d EMA50, volume > 1.5x 20-period average.
# Short when: price < Ichimoku cloud, Tenkan < Kijun, price < 1d EMA50, volume > 1.5x 20-period average.
# Exit when price crosses back into cloud or volume filter fails.
# Ichimoku components: Tenkan (9-period), Kijun (26-period), Senkou Span A/B (26-period, displaced 26 periods).
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Volume filter ensures participation and avoids low-conviction moves.
# Designed for 6h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
name = "6h_Ichimoku_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period)
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A and B (26-period, displaced 26 periods)
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = (pd.Series(high).rolling(window=52, min_periods=52).max().values + 
                pd.Series(low).rolling(window=52, min_periods=52).min().values) / 2
    
    # Displace Senkou Span A and B by 26 periods forward
    senkou_a_leading = np.full_like(senkou_a, np.nan)
    senkou_b_leading = np.full_like(senkou_b, np.nan)
    if len(senkou_a) >= 26:
        senkou_a_leading[26:] = senkou_a[:-26]
        senkou_b_leading[26:] = senkou_b[:-26]
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52 + 26  # Sufficient warmup for Ichimoku (52 for Senkou B, 26 for displacement)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a_leading[i]) or 
            np.isnan(senkou_b_leading[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = np.maximum(senkou_a_leading[i], senkou_b_leading[i])
        cloud_bottom = np.minimum(senkou_a_leading[i], senkou_b_leading[i])
        
        if position == 0:
            # Long conditions: price > cloud, Tenkan > Kijun, price > 1d EMA50, volume filter
            long_cond = (close[i] > cloud_top) and (tenkan[i] > kijun[i]) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price < cloud, Tenkan < Kijun, price < 1d EMA50, volume filter
            short_cond = (close[i] < cloud_bottom) and (tenkan[i] < kijun[i]) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below cloud top OR volume filter fails
            if close[i] < cloud_top or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud bottom OR volume filter fails
            if close[i] > cloud_bottom or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals