#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm
Hypothesis: Ichimoku cloud breakout with weekly trend filter and volume confirmation. 
In bull markets: price breaks above cloud with weekly uptrend → long. 
In bear markets: price breaks below cloud with weekly downtrend → short. 
Uses discrete sizing (0.25) and minimum holding period (4 bars) to reduce fee drag.
Target: 50-150 trades over 4 years. Works in both regimes by requiring alignment with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need 52 for weekly EMA26
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (6h timeframe)
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
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # Cloud: area between Senkou A and Senkou B
    # Bullish when price > Senkou Span A and Senkou Span A > Senkou Span B
    # Bearish when price < Senkou Span A and Senkou Span A < Senkou Span B
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Weekly EMA26 for trend filter
    ema_26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_26_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_26_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine cloud color and price position
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        close_val = close[i]
        
        # Bullish cloud: Senkou A > Senkou B
        # Bearish cloud: Senkou A < Senkou B
        cloud_bullish = senkou_a_val > senkou_b_val
        cloud_bearish = senkou_a_val < senkou_b_val
        
        # Price above/below cloud
        price_above_cloud = close_val > max(senkou_a_val, senkou_b_val)
        price_below_cloud = close_val < min(senkou_a_val, senkou_b_val)
        
        # Long logic: price breaks above bullish cloud with volume spike and weekly uptrend
        long_condition = price_above_cloud and cloud_bullish and volume_spike[i] and (close_val > ema_26_1w_aligned[i])
        # Short logic: price breaks below bearish cloud with volume spike and weekly downtrend
        short_condition = price_below_cloud and cloud_bearish and volume_spike[i] and (close_val < ema_26_1w_aligned[i])
        
        # Exit logic: price re-enters cloud or weekly trend reversal
        exit_long = close_val < min(senkou_a_val, senkou_b_val) or close_val < ema_26_1w_aligned[i]
        exit_short = close_val > max(senkou_a_val, senkou_b_val) or close_val > ema_26_1w_aligned[i]
        
        # Minimum holding period: 4 bars
        if position != 0 and bars_since_entry < 4:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0