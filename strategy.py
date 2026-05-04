#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Tenkan/Kijun cross and price position relative to Kumo (cloud) from 6h timeframe.
# Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts.
# Volume confirmation: current volume > 1.5x 20-period EMA of volume.
# Discrete sizing 0.25 limits risk. Target: 50-150 trades over 4 years (12-37/year).
# Works in bull/bear: trend filter aligns with higher timeframe direction, Ichimoku provides
# dynamic support/resistance and momentum signals effective in trending and ranging markets.

name = "6h_Ichimoku_1dEMA50_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Senkou spans are plotted 26 periods ahead, so we need to shift for current cloud
    senkou_a_shifted = senkou_a.shift(26)
    senkou_b_shifted = senkou_b.shift(26)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_shifted[i], senkou_b_shifted[i])
        cloud_bottom = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan/Kijun cross
        tenkan_kijun_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tenkan_kijun_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Long conditions: price above cloud + bullish TK cross + uptrend + volume spike
            if price_above_cloud and tenkan_kijun_cross_up and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below cloud + bearish TK cross + downtrend + volume spike
            elif price_below_cloud and tenkan_kijun_cross_down and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below cloud OR bearish TK cross OR trend changes
            if (price_below_cloud or 
                tenkan_kijun_cross_down or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud OR bullish TK cross OR trend changes
            if (price_above_cloud or 
                tenkan_kijun_cross_up or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals