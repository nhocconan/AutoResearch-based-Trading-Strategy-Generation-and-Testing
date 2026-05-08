#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud (Tenkan/Kijun) + 1d Cloud Filter + Volume Confirmation
# Uses Ichimoku for momentum and trend: Tenkan > Kijun for bullish, Tenkan < Kijun for bearish.
# Price above/below 1d Ichimoku Cloud confirms higher timeframe trend alignment.
# Volume spike (>2x 20-period average) filters for institutional participation.
# Targets 10-25 trades per year (~40-100 total over 4 years) to minimize fee drag.
# Designed to work in both bull (trend following) and bear (counter-trend reversals at cloud edges).

name = "6h_Ichimoku_1dCloud_Filter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku components (6-period Tenkan, 13-period Kijun, 26-period Senkou Span B)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Get 1d data for Ichimoku cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku Cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough data for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        senkou_a_1d = senkou_a_1d_aligned[i]
        senkou_b_1d = senkou_b_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        # Determine 6h Ichimoku trend
        bullish_6h = tenkan_val > kijun_val
        bearish_6h = tenkan_val < kijun_val
        
        # Determine price position relative to 6h cloud
        above_cloud_6h = close[i] > max(senkou_a_val, senkou_b_val)
        below_cloud_6h = close[i] < min(senkou_a_val, senkou_b_val)
        
        # Determine price position relative to 1d cloud
        above_cloud_1d = close[i] > max(senkou_a_1d, senkou_b_1d)
        below_cloud_1d = close[i] < min(senkou_a_1d, senkou_b_1d)
        
        if position == 0:
            # Enter long: 6h bullish TK cross, price above both clouds, volume confirmation
            if bullish_6h and above_cloud_6h and above_cloud_1d and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: 6h bearish TK cross, price below both clouds, volume confirmation
            elif bearish_6h and below_cloud_6h and below_cloud_1d and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 6h bearish TK cross OR price drops below 6h cloud
            if bearish_6h or below_cloud_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 6h bullish TK cross OR price rises above 6h cloud
            if bullish_6h or above_cloud_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals