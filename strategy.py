#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dTrend_Filter_VolumeConfirm
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d trend filter (price > EMA50) and volume confirmation (>1.5x 20-bar avg). Enters long when TK crosses up in 1d uptrend, short when crosses down in 1d downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 6h timeframe with ~12-30 trades/year, works in bull/bear by following 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 26-period data for Kijun and 50 for 1d EMA
    start_idx = max(26, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK crosses up (tenkan > kijun) in 1d uptrend with volume confirmation
            tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            bullish_condition = tk_cross_up and (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            
            # Short: TK crosses down (tenkan < kijun) in 1d downtrend with volume confirmation
            tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            bearish_condition = tk_cross_down and (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
            if bullish_condition:
                signals[i] = 0.25
                position = 1
            elif bearish_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TK crosses down OR trend turns down
            tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            if tk_cross_down or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK crosses up OR trend turns up
            tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            if tk_cross_up or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dTrend_Filter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0