#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_TK_Cross_1dTrend
Hypothesis: On 6h timeframe, enter long when Tenkan-Sen crosses above Kijun-Sen AND price is above Kumo (cloud) from 1d timeframe AND 1d trend is up (close > EMA50). Enter short when Tenkan-Sen crosses below Kijun-Sen AND price is below Kumo AND 1d trend is down. Uses Ichimoku for momentum and trend confirmation, with 1d timeframe for higher timeframe trend filter and cloud support/resistance. Designed to generate ~10-20 trades per year on BTC/ETH/SOL with Sharpe > 0 in both bull and bear regimes by avoiding whipsaws via cloud filter and higher timeframe alignment.
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
    
    # Get 1d data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need enough for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Kumo (cloud) boundaries: Senkou Span A and B shifted 26 periods back
    # So we use values from 26 periods ago to represent current cloud
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    # First 26 values are invalid
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_current)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_current)
    
    # 1d trend filter: EMA50
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup (52) + EMA warmup (50) + alignment offset
    start_idx = 78  # 52 + 26 for cloud shift safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        # Price above cloud: price > both Senkou Span A and B
        price_above_cloud = close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]
        # Price below cloud: price < both Senkou Span A and B
        price_below_cloud = close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: TK cross up + price above cloud + 1d uptrend
            long_signal = tk_cross_up and price_above_cloud and trend_uptrend
            # Short: TK cross down + price below cloud + 1d downtrend
            short_signal = tk_cross_down and price_below_cloud and trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR price falls below cloud OR trend change to downtrend
            if tk_cross_down or not price_above_cloud or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud OR trend change to uptrend
            if tk_cross_up or not price_below_cloud or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0