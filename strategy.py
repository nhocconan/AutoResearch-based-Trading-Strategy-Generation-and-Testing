#!/usr/bin/env python3
# 6h_ichimoku_cloud_regime_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d HTF for trend direction (price above/below cloud) and TK cross from 6h for entry timing. Volume confirmation (>1.3x 20-bar avg) filters weak breakouts. Works in bull/bear: cloud acts as dynamic support/resistance, TK cross captures momentum shifts, volume ensures conviction. Discrete sizing (0.25) limits drawdown in crashes. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h TK Cross components (9-period)
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tk_line = (high_9 + low_9) / 2
    
    # 6h Kijun Base Line (26-period)
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_line = (high_26 + low_26) / 2
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d Ichimoku Cloud
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (Conversion Line) - 9-period
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    # 1d Kijun-sen (Base Line) - 26-period
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # 1d Senkou Span A (Leading Span A) - (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B (Leading Span B) - (52-period high+low)/2 plotted 26 periods ahead
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Align all 1d components to 6h timeframe (with proper delay for completed candles)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tk_line[i]) or np.isnan(kijun_line[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        # TK Cross signals
        tk_cross_above = tk_line[i] > kijun_line[i] and tk_line[i-1] <= kijun_line[i-1]
        tk_cross_below = tk_line[i] < kijun_line[i] and tk_line[i-1] >= kijun_line[i-1]
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR TK cross below
            if close[i] < cloud_bottom[i] or tk_cross_below:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR TK cross above
            if close[i] > cloud_top[i] or tk_cross_above:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: TK cross above + price above cloud + volume confirmation
            if tk_cross_above and price_above_cloud and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short: TK cross below + price below cloud + volume confirmation
            elif tk_cross_below and price_below_cloud and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals