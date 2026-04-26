#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_v1
Hypothesis: 6h Ichimoku signals (TK cross) filtered by 1d cloud direction and volume spike.
- Long when Tenkan crosses above Kijun on 6h AND price > 1d Kumo cloud AND volume > 1.5x 20-period avg
- Short when Tenkan crosses below Kijun on 6h AND price < 1d Kumo cloud AND volume > 1.5x 20-period avg
- Uses Ichimoku from completed 6h bars for entry timing, 1d cloud for trend filter
- Volume spike confirms institutional participation
- Designed for moderate frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite TK cross or cloud penetration
- Novelty: Combines Ichimoku momentum with higher timeframe trend filter and volume confirmation for BTC/ETH edge
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Ichimoku components on 6h (using completed bars only)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan = (pd.Series(df_6h['high'].values).rolling(window=period_tenkan, min_periods=period_tenkan).max() +
              pd.Series(df_6h['low'].values).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun = (pd.Series(df_6h['high'].values).rolling(window=period_kijun, min_periods=period_kijun).max() +
             pd.Series(df_6h['low'].values).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = ((pd.Series(df_6h['high'].values).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() +
                 pd.Series(df_6h['low'].values).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed for structure)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b.values)
    
    # Load daily data ONCE before loop for cloud trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku cloud (Senkou Span A and B)
    # Tenkan-sen (1d): (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max() +
                 pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (1d): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max() +
                pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (1d): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(26)
    
    # Senkou Span B (1d): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_1d = ((pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max() +
                    pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align daily cloud to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Determine cloud top and bottom (cloud is between Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Cloud trend: 1 = bullish (price above cloud), -1 = bearish (price below cloud), 0 = inside cloud
    cloud_trend = np.where(close > cloud_top, 1,
                          np.where(close < cloud_bottom, -1, 0))
    
    # Volume spike filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # TK cross signals: 1 = bullish cross (Tenkan > Kijun), -1 = bearish cross (Tenkan < Kijun)
    tk_cross = np.where(tenkan_aligned > kijun_aligned, 1,
                       np.where(tenkan_aligned < kijun_aligned, -1, 0))
    
    # Detect TK cross changes (actual crossovers)
    tk_cross_change = np.diff(tk_cross, prepend=tk_cross[0])
    tk_bullish_cross = (tk_cross_change == 2)  # -1 to 1
    tk_bearish_cross = (tk_cross_change == -2)  # 1 to -1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 20 for volume MA)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions with cloud trend and volume spike filter
        if position == 0:
            # Long: Bullish TK cross AND price above cloud AND volume spike
            if tk_bullish_cross[i] and cloud_trend[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross AND price below cloud AND volume spike
            elif tk_bearish_cross[i] and cloud_trend[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bearish TK cross OR price penetrates cloud (below cloud bottom)
            if tk_bearish_cross[i] or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bullish TK cross OR price penetrates cloud (above cloud top)
            if tk_bullish_cross[i] or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0