#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d EMA50 trend filter and volume confirmation.
- Long: Tenkan > Kijun AND price > Cloud (Senkou Span A/B) AND price > 1d EMA50 AND volume > 1.5x 20-period avg
- Short: Tenkan < Kijun AND price < Cloud AND price < 1d EMA50 AND volume > 1.5x 20-period avg
- Exit: Opposite Ichimoku signal OR price crosses 1d EMA50
- Ichimoku provides institutional support/resistance levels; works in bull (buy cloud breakouts) and bear (sell cloud breakdowns)
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Ichimoku Cloud (9, 26, 52) - Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B: (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # The actual cloud boundaries for current price (shifted back by 26)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(78, 20)  # 52+26 for Senkou Span, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(tenkan[i]) or
            np.isnan(kijun[i]) or
            np.isnan(senkou_a_shifted[i]) or
            np.isnan(senkou_b_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Ichimoku signals
        bullish_ichimoku = tenkan[i] > kijun[i] and close[i] > max(senkou_a_shifted[i], senkou_b_shifted[i])
        bearish_ichimoku = tenkan[i] < kijun[i] and close[i] < min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Long: Bullish Ichimoku AND price > 1d EMA50 AND volume confirmation
            if bullish_ichimoku and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Ichimoku AND price < 1d EMA50 AND volume confirmation
            elif bearish_ichimoku and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Ichimoku OR price < 1d EMA50 (trend flip)
            if bearish_ichimoku or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Ichimoku OR price > 1d EMA50 (trend flip)
            if bullish_ichimoku or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0