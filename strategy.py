#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud Breakout with 1d TK Cross Filter and Volume Confirmation
- Ichimoku Cloud (Senkou Span A/B) from 6h provides dynamic support/resistance
- 1d Tenkan-Kijun (TK) cross determines higher timeframe trend bias
- Volume > 1.8x 20-period average confirms breakout strength
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in bull markets via cloud breakouts with TK cross up, in bear markets via breakdowns with TK cross down
- Uses actual Ichimoku formulas: Tenkan = (9-period high + low)/2, Kijun = (26-period high + low)/2,
  Senkou A = (Tenkan + Kijun)/2 shifted 26 ahead, Senkou B = (52-period high + low)/2 shifted 26 ahead
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TK cross trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Get 1d TK cross for trend filter
    # Tenkan-sen 1d: (9-period high + low) / 2
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d: (26-period high + low) / 2
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # TK cross: 1 = bullish (Tenkan > Kijun), -1 = bearish (Tenkan < Kijun)
    tk_cross_1d = np.where(tenkan_1d > kijun_1d, 1, np.where(tenkan_1d < kijun_1d, -1, 0))
    
    # Align indicators to 6h timeframe (completed 1d bar only)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    tk_cross_1d_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 20)  # Senkou B, Kijun, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(tk_cross_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Ichimoku signals with TK cross filter and volume confirmation
        # Long: price breaks above cloud + bullish TK cross + volume spike
        # Short: price breaks below cloud + bearish TK cross + volume spike
        long_signal = (close[i] > cloud_top and 
                      tk_cross_1d_aligned[i] == 1 and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < cloud_bottom and 
                       tk_cross_1d_aligned[i] == -1 and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price re-enters cloud or opposite TK cross
            exit_signal = False
            
            if position == 1:
                # Exit long: price re-enters cloud or bearish TK cross
                if (close[i] < cloud_top or 
                    tk_cross_1d_aligned[i] == -1):
                    exit_signal = True
            elif position == -1:
                # Exit short: price re-enters cloud or bullish TK cross
                if (close[i] > cloud_bottom or 
                    tk_cross_1d_aligned[i] == 1):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_Volume"
timeframe = "6h"
leverage = 1.0