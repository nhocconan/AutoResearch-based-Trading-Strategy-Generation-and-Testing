#!/usr/bin/env python3
"""
Hypothesis: 6h 12h Ichimoku Cloud with 1d trend filter for BTC/ETH.
Uses 12h Ichimoku (TK cross + cloud filter) for entry timing, 1d EMA50 for trend direction, 
and volume confirmation on breakouts. Long when TK crosses above KJ and price above cloud 
in uptrend (price > 1d EMA50) with volume spike. Short when TK crosses below KJ and 
price below cloud in downtrend (price < 1d EMA50) with volume spike. 
Ichimoku provides dynamic support/resistance and trend strength, reducing whipsaw in 
bear markets. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Ichimoku
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_12h).rolling(window=9, min_periods=9).max().values + 
                  pd.Series(low_12h).rolling(window=9, min_periods=9).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_12h).rolling(window=26, min_periods=26).max().values + 
                 pd.Series(low_12h).rolling(window=26, min_periods=26).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (pd.Series(high_12h).rolling(window=52, min_periods=52).max().values + 
                     pd.Series(low_12h).rolling(window=52, min_periods=52).min().values) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    
    # Get 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume spike (volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.8)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross signals
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = tk_cross_up and price_above_cloud and uptrend and vol_confirm
        short_entry = tk_cross_down and price_below_cloud and downtrend and vol_confirm
        
        # Exit when TK cross reverses or price exits cloud in opposite direction
        exit_long = (position == 1 and 
                    (tk_cross_down or 
                     (price_below_cloud and close[i] < kijun_sen_aligned[i])))
        exit_short = (position == -1 and 
                     (tk_cross_up or 
                      (price_above_cloud and close[i] > kijun_sen_aligned[i])))
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_ichimoku_1d_trend"
timeframe = "6h"
leverage = 1.0