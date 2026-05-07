#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Tenkan_Kijun_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_1d, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need at least 26 days for Ichimoku
        return np.zeros(n)
    
    # Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Daily trend filter: price above/below Kijun
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    trend_up = close_1d_aligned > kijun_aligned
    trend_down = close_1d_aligned < kijun_aligned
    
    # Volume filter: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1.5 days for 6h
    
    start_idx = max(30, 52)  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(trend_up[i]) or 
            np.isnan(trend_down[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Tenkan crosses above Kijun, price above cloud, in uptrend, with volume
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1] and 
                close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i]) and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Tenkan crosses below Kijun, price below cloud, in downtrend, with volume
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  tenkan_aligned[i-1] >= kijun_aligned[i-1] and 
                  close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i]) and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Tenkan crosses below Kijun OR price below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] and 
                tenkan_aligned[i-1] >= kijun_aligned[i-1]) or \
               close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun OR price above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1]) or \
               close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku system on daily timeframe for trend and cloud support/resistance,
# with Tenkan/Kijun cross on 6h for entry timing. Works in bull markets by capturing
# trend continuation above the cloud, and in bear markets by capturing
# trend reversals below the cloud. Volume filter reduces false signals.
# Tenkan/Kijun cross provides timely entries while cloud acts as dynamic support/resistance.
# Target: 15-30 trades/year to avoid fee drift. Ichimoku is proven effective
# on daily charts for major cryptocurrencies. The cloud filter ensures we only
# trade in the direction of the higher timeframe trend.