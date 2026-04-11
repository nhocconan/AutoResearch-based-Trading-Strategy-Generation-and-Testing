#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Long: Price above 1d Kumo (cloud), Tenkan > Kijun on 6h, volume > 1.3x 20-period average
# - Short: Price below 1d Kumo (cloud), Tenkan < Kijun on 6h, volume > 1.3x 20-period average
# - Exit: Price crosses opposite Tenkan-Kijun line on 6h
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Ichimoku provides dynamic support/resistance and trend direction
# - 1d cloud filter ensures alignment with higher timeframe trend
# - Volume confirmation filters out weak signals

name = "6h_1d_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return signals
    
    # Pre-compute 1d Ichimoku cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high + period52_low) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe (with proper delay for completed bars)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Pre-compute 6h Ichimoku components for entry signals
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(volume_sma_20_6h[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # 6h Ichimoku signals
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        
        # 1d Cloud boundaries (Senkou Span A and B)
        senkou_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        senkou_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20_6h[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price above 1d cloud, Tenkan > Kijun on 6h, volume confirmation
        if (close_price > senkou_top and 
            tenkan_6h_val > kijun_6h_val and 
            vol_confirm):
            enter_long = True
        
        # Short: Price below 1d cloud, Tenkan < Kijun on 6h, volume confirmation
        if (close_price < senkou_bottom and 
            tenkan_6h_val < kijun_6h_val and 
            vol_confirm):
            enter_short = True
        
        # Exit conditions: Tenkan-Kijun cross in opposite direction
        exit_long = (tenkan_6h_val < kijun_6h_val)
        exit_short = (tenkan_6h_val > kijun_6h_val)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals