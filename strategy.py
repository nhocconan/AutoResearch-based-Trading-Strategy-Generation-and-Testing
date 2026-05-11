# 12h_1d_Ichimoku_TenkanKijun_Cross_v1
# Hypothesis: Uses daily Ichimoku Tenkan-Kijun cross as entry signal with weekly price relative to cloud as trend filter.
# Long when price above weekly cloud and daily Tenkan crosses above Kijun.
# Short when price below weekly cloud and daily Tenkan crosses below Kijun.
# Includes volume confirmation to filter false signals.
# Designed for low trade frequency (12-37/year) via weekly trend filter and daily entry signal.
# Works in both bull and bear markets by following higher-timeframe trend.

name = "12h_1d_Ichimoku_TenkanKijun_Cross_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(26)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Ichimoku for Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Align weekly Ichimoku to 12h timeframe
    tenkan_1w_12h = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_12h = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_12h = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_12h = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    chikou_1w_12h = align_htf_to_ltf(prices, df_1w, chikou_1w)
    
    # --- Daily Ichimoku for Entry Signal ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, _, _, _ = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align daily Ichimoku to 12h timeframe
    tenkan_1d_12h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_12h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    # Calculate Tenkan-Kijun cross signals
    tk_cross_above = (tenkan_1d_12h > kijun_1d_12h) & (tenkan_1d_12h <= kijun_1d_12h)
    tk_cross_below = (tenkan_1d_12h < kijun_1d_12h) & (tenkan_1d_12h >= kijun_1d_12h)
    
    # --- Volume Spike Detection (24-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1w_12h[i]) or np.isnan(kijun_1w_12h[i]) or 
            np.isnan(senkou_a_1w_12h[i]) or np.isnan(senkou_b_1w_12h[i]) or
            np.isnan(tenkan_1d_12h[i]) or np.isnan(kijun_1d_12h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        green_cloud = senkou_a_1w_12h[i] > senkou_b_1w_12h[i]
        red_cloud = senkou_a_1w_12h[i] < senkou_b_1w_12h[i]
        
        above_cloud = close[i] > max(senkou_a_1w_12h[i], senkou_b_1w_12h[i])
        below_cloud = close[i] < min(senkou_a_1w_12h[i], senkou_b_1w_12h[i])
        in_cloud = not above_cloud and not below_cloud
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price above green cloud + TK cross up + volume
            if (above_cloud and green_cloud and 
                tk_cross_above[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below red cloud + TK cross down + volume
            elif (below_cloud and red_cloud and 
                  tk_cross_below[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite cloud break or TK cross in opposite direction
            if position == 1:
                # Exit long: price breaks below cloud OR TK cross down
                if below_cloud or tk_cross_below[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above cloud OR TK cross up
                if above_cloud or tk_cross_above[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals