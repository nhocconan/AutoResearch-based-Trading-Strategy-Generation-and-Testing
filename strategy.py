#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike
Hypothesis: Use 1d Ichimoku cloud as trend filter (price above/below cloud) on 6h timeframe with volume spike confirmation for breakout entries.
Ichimoku cloud provides dynamic support/resistance and trend direction that adapts to volatility.
Volume spike ensures momentum confirmation. Targets 12-30 trades/year by requiring: 1) price breaks 6h Donchian(20) in direction of 1d Ichimoku trend, 2) volume > 2.0x 20-period average.
Works in bull/bear: cloud acts as dynamic trend filter, volume spike avoids false breakouts.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displaced)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 displaced 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # The cloud is between Senkou Span A and B
    # Current cloud values (displaced back 26 periods to align with current price)
    # We need values from 26 periods ago to represent current cloud
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_lagged)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_lagged)
    
    # 6h Donchian(20) for breakout levels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) and Donchian (20)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Ichimoku trend: price above/below cloud
        top_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        bottom_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = curr_close > top_cloud
        price_below_cloud = curr_close < bottom_cloud
        
        # Cloud color (green/red): Senkou A > Senkou B = bullish cloud
        bullish_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with bullish cloud alignment and volume
            long_breakout = (curr_high > period20_high[i]) and price_above_cloud and bullish_cloud and volume_confirm[i]
            # Short breakout: price breaks below Donchian low with bearish cloud alignment and volume
            short_breakout = (curr_low < period20_low[i]) and price_below_cloud and (not bullish_cloud) and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price breaks below cloud or Donchian low
            if curr_close < bottom_cloud or curr_low < period20_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above cloud or Donchian high
            if curr_close > top_cloud or curr_high > period20_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0