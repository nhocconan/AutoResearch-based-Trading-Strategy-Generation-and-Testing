#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeConfirm
Hypothesis: 6h Ichimoku cloud breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND price > 12h EMA50 AND volume spike.
Short when price breaks below Ichimoku cloud AND price < 12h EMA50 AND volume spike.
Exit when price re-enters the cloud or loses 12h EMA50 alignment.
Ichimoku provides dynamic support/resistance and trend direction, effective in both bull and bear markets.
Designed for 12-30 trades/year on 6h to minimize fee drag while capturing strong breaks aligned with 12h trend.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # Upper cloud boundary = max(Senkou Span A, Senkou Span B)
    # Lower cloud boundary = min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align Ichimoku cloud to 6h (already on 6h, no alignment needed)
    # But we need to shift Senkou Span B by 26 periods forward (it's plotted 26 periods ahead)
    # For breakout detection, we use the current cloud values (already shifted in calculation)
    # Actually, Senkou Span A and B are plotted 26 periods ahead, so we need to use values from 26 periods ago
    # For simplicity and to avoid look-ahead, we'll use the current calculated cloud as support/resistance
    # This is acceptable as the cloud represents future support/resistance based on past data
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Ichimoku (52 periods) + 12h EMA50 + volume avg
    start_idx = max(52, 60, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Cloud breakout with 12h EMA50 alignment and volume spike
            # Long: Close > Upper Cloud AND price > 12h EMA50 AND volume spike
            # Short: Close < Lower Cloud AND price < 12h EMA50 AND volume spike
            long_condition = (close_val > upper_cloud_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < lower_cloud_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price re-enters cloud (below upper cloud) OR loses 12h EMA50 alignment
            if close_val < upper_cloud_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters cloud (above lower cloud) OR loses 12h EMA50 alignment
            if close_val > lower_cloud_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0