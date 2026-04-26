#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeFilter
Hypothesis: Ichimoku cloud (TK cross + price above/below cloud) on 6h timeframe with 1d trend filter (price > EMA50) and volume confirmation (>1.5x 20-bar MA). 
Ichimoku provides dynamic support/resistance and trend direction, reducing whipsaws in ranging markets. 
Volume filter ensures breakouts have conviction. 
Target: 12-30 trades/year on BTC/ETH by requiring confluence of multiple conditions.
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): not used for signals to avoid look-ahead
    
    # Align Ichimoku components (already on 6h timeframe)
    tenkan_aligned = tenkan
    kijun_aligned = kijun
    senkou_a_aligned = senkou_a
    senkou_b_aligned = senkou_b
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size as fraction of capital
    
    # Warmup: max of calculations (26 for Kijun, 52 for Senkou B, 20 for volume MA)
    start_idx = max(26, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Ichimoku signals:
        # Bullish: Tenkan > Kijun AND price above cloud
        # Bearish: Tenkan < Kijun AND price below cloud
        bullish_ichimoku = (tenkan_val > kijun_val) and (close_val > upper_cloud)
        bearish_ichimoku = (tenkan_val < kijun_val) and (close_val < lower_cloud)
        
        # Entry conditions: Ichimoku signal in direction of 1d trend with volume confirmation
        long_entry = bullish_ichimoku and bullish_1d and vol_spike
        short_entry = bearish_ichimoku and bearish_1d and vol_spike
        
        # Exit conditions: opposite Ichimoku signal or trend reversal
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (bearish_ichimoku or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bullish_ichimoku or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0