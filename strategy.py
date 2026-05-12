#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Signal_1dTrend
# Hypothesis: On 6h timeframe, use Ichimoku Cloud from daily timeframe for trend direction,
# and Tenkan-Kijun cross from 6h for entry timing. Enter long when price is above daily cloud
# and Tenkan crosses above Kijun. Enter short when price is below daily cloud and Tenkan crosses below Kijun.
# Exit when price crosses back into the cloud or Tenkan-Kijun reverses.
# Uses daily cloud as major trend filter to avoid counter-trend trades, targeting 20-40 trades/year.
# Works in bull via cloud support and in bear via cloud resistance with alignment.

name = "6h_Ichimoku_Cloud_Signal_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data for Ichimoku Cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(daily_high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(daily_low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(daily_high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(daily_low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(daily_high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(daily_low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Shift Senkou spans forward by 26 periods (for cloud ahead)
    # But for cloud filter, we need current cloud, so we use unshifted spans
    # The cloud is between Senkou A and Senkou B at current time
    
    # Align Ichimoku components to 6h timeframe (with 1-bar delay for completed daily bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6h Tenkan-Kijun for entry signals
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2.0
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2.0
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou A and B)
        top_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        bottom_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        vol_confirm = volume_confirm[i]
        
        # Check for Tenkan-Kijun cross on 6h
        tk_cross_up = tenkan_6h_val > kijun_6h_val and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h_val < kijun_6h_val and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # LONG: Price above cloud, Tenkan crosses above Kijun, with volume confirmation
            if close[i] > top_cloud and tk_cross_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud, Tenkan crosses below Kijun, with volume confirmation
            elif close[i] < bottom_cloud and tk_cross_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below cloud or Tenkan crosses below Kijun
            if close[i] < top_cloud or (tenkan_6h_val < kijun_6h_val and tenkan_6h[i-1] >= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above cloud or Tenkan crosses above Kijun
            if close[i] > bottom_cloud or (tenkan_6h_val > kijun_6h_val and tenkan_6h[i-1] <= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals