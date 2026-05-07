#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Uses Ichimoku components from daily data (Tenkan, Kijun, Senkou A/B) to identify trend direction
# and cloud support/resistance. Enters on 6h breakout of cloud in direction of 1d trend, 
# confirmed by volume spike. Works in both bull/bear markets by following 1d trend.
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid excessive fee drag.
name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
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
    
    # Load 1d data ONCE for Ichimoku calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_period = 9
    max_high_9 = pd.Series(high_1d).rolling(window=tenkan_period, center=False).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=tenkan_period, center=False).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_period = 26
    max_high_26 = pd.Series(high_1d).rolling(window=kijun_period, center=False).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=kijun_period, center=False).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_period = 52
    max_high_52 = pd.Series(high_1d).rolling(window=senkou_b_period, center=False).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=senkou_b_period, center=False).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1d trend filter: price above/below Kijun-sen
    # Use Senkou Span A/B to determine cloud top/bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for Ichimoku calculation
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below cloud (using Kijun as secondary filter)
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        price_in_cloud = ~(price_above_cloud | price_below_cloud)
        
        # Additional trend filter: price above/below Kijun-sen
        price_above_kijun = close[i] > kijun_aligned[i]
        price_below_kijun = close[i] < kijun_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above cloud with volume spike and bullish alignment
            long_condition = price_above_cloud and vol_spike[i] and price_above_kijun
            # Short breakdown: price breaks below cloud with volume spike and bearish alignment
            short_condition = price_below_cloud and vol_spike[i] and price_below_kijun
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters cloud or turns bearish
            if price_in_cloud or price_below_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters cloud or turns bullish
            if price_in_cloud or price_above_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals