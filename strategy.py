#!/usr/bin/env python3

"""
Hypothesis: 6-hour Ichimoku Cloud with Tenkan/Kijun cross + weekly trend filter.
Trades Tenkan/Kijun cross in the direction of weekly EMA34 trend.
Uses cloud color for additional filter and volatility-based exits.
Designed for low trade frequency (12-37/year) to minimize fee drift and work in both bull and bear markets
by aligning with higher timeframe trend and using cloud as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud color: green (bullish) when Senkou A > Senkou B
        cloud_green = senkou_a_aligned[i] > senkou_b_aligned[i]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, in uptrend, above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1] and
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and
                close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, in downtrend, below cloud
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  tenkan_aligned[i-1] >= kijun_aligned[i-1] and
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and
                  close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Tenkan/Kijun cross in opposite direction or price breaks cloud in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: Tenkan crosses below Kijun OR price falls below cloud
                if (tenkan_aligned[i] < kijun_aligned[i] and 
                    tenkan_aligned[i-1] >= kijun_aligned[i-1]) or \
                   (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Tenkan crosses above Kijun OR price rises above cloud
                if (tenkan_aligned[i] > kijun_aligned[i] and 
                    tenkan_aligned[i-1] <= kijun_aligned[i-1]) or \
                   (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wEMA34_Trend"
timeframe = "6h"
leverage = 1.0