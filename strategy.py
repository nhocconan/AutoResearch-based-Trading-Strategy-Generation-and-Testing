#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6-hour Ichimoku Cloud with 12-hour trend filter and volume confirmation
    # Uses Tenkan/Kijun cross + price relative to cloud for entry/exit
    # 12-hour EMA50 filters trend direction to avoid counter-trend trades
    # Volume spike confirms institutional participation
    # Targets ~20-30 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_6h).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low_6h).rolling(window=9, min_periods=9).min()
    tenkan_6h = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_6h).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low_6h).rolling(window=26, min_periods=26).min()
    kijun_6h = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_6h = (tenkan_6h + kijun_6h) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_6h).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low_6h).rolling(window=52, min_periods=52).min()
    senkou_b_6h = (high_52 + low_52) / 2
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to main timeframe
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h.values)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h.values)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h.values)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h.values)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or
            np.isnan(senkou_a_6h_aligned[i]) or np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        cloud_bottom = min(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        
        if position == 0:
            # Long: Tenkan > Kijun (bullish cross) + price above cloud + above 12h EMA50 + volume spike
            if (tenkan_6h_aligned[i] > kijun_6h_aligned[i] and 
                close[i] > cloud_top and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun (bearish cross) + price below cloud + below 12h EMA50 + volume spike
            elif (tenkan_6h_aligned[i] < kijun_6h_aligned[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Tenkan/Kijun cross reverses or price enters cloud
            if position == 1:
                if (tenkan_6h_aligned[i] < kijun_6h_aligned[i] or 
                    close[i] < cloud_top):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (tenkan_6h_aligned[i] > kijun_6h_aligned[i] or 
                    close[i] > cloud_bottom):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_12hEMA50_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0