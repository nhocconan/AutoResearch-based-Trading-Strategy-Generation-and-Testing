#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND price > 1d EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below Ichimoku cloud AND price < 1d EMA50 AND volume > 1.8x 20-bar avg
# Exit when price re-enters the cloud (between Senkou Span A and B)
# Uses Ichimoku from 6h timeframe with 1d EMA50 trend filter to avoid counter-trend trades
# Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to avoid overtrading
# Works in bull markets by capturing upside breaks above cloud and in bear markets by shorting breakdowns below cloud
# with trend alignment ensuring trades follow higher timeframe momentum.

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period_9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period_9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period_9_high + period_9_low) / 2.0
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period_26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period_26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period_26_high + period_26_low) / 2.0
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period_52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period_52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period_52_high + period_52_low) / 2.0
    
    # The cloud is between Senkou Span A and B
    # Upper cloud boundary = max(Senkou Span A, Senkou Span B)
    # Lower cloud boundary = min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 52, 20)  # Base Line, Span B, and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_upper_cloud = upper_cloud[i]
        curr_lower_cloud = lower_cloud[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price re-enters the cloud (between upper and lower cloud)
            if curr_close <= curr_upper_cloud and curr_close >= curr_lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters the cloud (between upper and lower cloud)
            if curr_close <= curr_upper_cloud and curr_close >= curr_lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper cloud AND price > 1d EMA50 AND volume confirmation
            if curr_close > curr_upper_cloud and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower cloud AND price < 1d EMA50 AND volume confirmation
            elif curr_close < curr_lower_cloud and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals