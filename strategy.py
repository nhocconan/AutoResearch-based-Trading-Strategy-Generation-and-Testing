#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h for entry signals.
# 1d EMA50 determines higher timeframe trend - only take longs in uptrend, shorts in downtrend.
# Volume spike (>1.5x 20-bar average) confirms breakout strength.
# Cloud acts as dynamic support/resistance - price must break above/below cloud for entry.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "6h_Ichimoku_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    lowest_low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    lowest_low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    highest_high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    lowest_low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Current cloud boundaries (Senkou Span A/B shifted back 26 periods to align with price)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # 6h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need 52 periods for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan_sen[i]) or
            np.isnan(kijun_sen[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 direction
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Ichimoku conditions
        bullish_tk_cross = tenkan_sen[i] > kijun_sen[i]
        bearish_tk_cross = tenkan_sen[i] < kijun_sen[i]
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Long entry: price above cloud + bullish TK cross + uptrend + volume
        long_entry = price_above_cloud and bullish_tk_cross and price_above_ema and vol_confirm
        # Short entry: price below cloud + bearish TK cross + downtrend + volume
        short_entry = price_below_cloud and bearish_tk_cross and price_below_ema and vol_confirm
        
        # Exit: TK cross in opposite direction OR price re-enters cloud
        long_exit = (not bullish_tk_cross) or (close[i] < cloud_top[i])
        short_exit = (not bearish_tk_cross) or (close[i] > cloud_bottom[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals