#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h for entry signals,
# 1d EMA50 for trend filter to avoid counter-trend trades, and volume spike for confirmation.
# Designed for low trade frequency (~12-30/year) to minimize fee drag on 6h timeframe.
# Ichimoku cloud acts as dynamic support/resistance - price above cloud = bullish bias,
# below cloud = bearish bias. TK cross provides momentum signals within the trend.
# Works in both bull and bear markets by only taking trades aligned with 1d trend.

name = "6h_Ichimoku_Cloud_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
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
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for entry as it requires future data
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # warmup for Senkou Span B (52) and Kijun (26)
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_cloud_top = cloud_top[i]
        curr_cloud_bottom = cloud_bottom[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Determine if price is above or below cloud
        price_above_cloud = curr_close > curr_cloud_top
        price_below_cloud = curr_close < curr_cloud_bottom
        
        # TK cross signals
        tk_cross_up = curr_tenkan > curr_kijun and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = curr_tenkan < curr_kijun and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long: price above cloud, TK cross up, uptrend (price > 1d EMA50), volume spike
            if price_above_cloud and tk_cross_up and curr_close > curr_ema_50 and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK cross down, downtrend (price < 1d EMA50), volume spike
            elif price_below_cloud and tk_cross_down and curr_close < curr_ema_50 and curr_volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit if price closes below cloud (trend change) or TK cross down
            if curr_close < curr_cloud_bottom or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit if price closes above cloud (trend change) or TK cross up
            if curr_close > curr_cloud_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals