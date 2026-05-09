#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud system with 1d trend filter and volume confirmation
# Long when price is above Kumo (cloud), Tenkan > Kijun, and 1d EMA50 uptrend with volume spike
# Short when price is below Kumo, Tenkan < Kijun, and 1d EMA50 downtrend with volume spike
# Exit when price crosses back into Kumo or Tenkan/Kijun cross reverses
# Ichimoku provides dynamic support/resistance and trend direction, suitable for 6b timeframe
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_Ichimoku_Cloud_1dEMA50_Volume"
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
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # But for signal generation, we use current cloud (Senkou spans shifted 26 periods back)
    # So we shift Senkou spans forward by 26 to get the cloud ahead, then compare price to it
    # For cloud at current period, we use Senkou spans calculated 26 periods ago
    senkou_a_shifted = senkou_a.shift(26)
    senkou_b_shifted = senkou_b.shift(26)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for Ichimoku calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan.iloc[i]) or np.isnan(kijun.iloc[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan_val = tenkan.iloc[i]
        kijun_val = kijun.iloc[i]
        
        if position == 0:
            # Enter long: price above cloud, Tenkan > Kijun, 1d EMA50 up, volume spike
            if (close[i] > cloud_top[i] and 
                tenkan_val > kijun_val and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud, Tenkan < Kijun, 1d EMA50 down, volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan_val < kijun_val and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses into cloud or Tenkan/Kijun cross reverses
            if (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]) or (tenkan_val < kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses into cloud or Tenkan/Kijun cross reverses
            if (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]) or (tenkan_val > kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals