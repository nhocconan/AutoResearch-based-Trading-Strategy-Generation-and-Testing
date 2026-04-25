#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Trend Filter and Volume Spike
Hypothesis: Ichimoku cloud (Tenkan/Kijun cross + cloud) provides high-probability entries in both bull and bear markets.
Using 1d timeframe for cloud calculation and trend filter (price above/below cloud) reduces false signals.
Volume spike confirms momentum. Targets 12-37 trades/year on 6h to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku and trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Current cloud: Senkou A and B shifted back 26 periods (to align with current price)
    # We need to shift the calculated Senkou A/B back by 26 to get today's cloud
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_lagged)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_lagged)
    
    # 1d EMA 50 for trend filter (only needs completed 1d candle)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume MA for 6h volume spike
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku calculation (52 for Senkou B) + volume MA
    start_idx = max(52 + 26, 20)  # 52 for Senkou B calc + 26 shift for cloud, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Cloud boundaries: top is max of Senkou A/B, bottom is min
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Ichimoku signals:
        # Bullish: Tenkan crosses above Kijun AND price above cloud
        # Bearish: Tenkan crosses below Kijun AND price below cloud
        # We use the current bar values (no look-ahead)
        bullish_cross = tenkan_val > kijun_val
        bearish_cross = tenkan_val < kijun_val
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Volume confirmation: current 6h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_6h
        
        if position == 0:
            # Look for entry signals
            # Long: bullish TK cross + price above cloud + above 1d EMA + volume confirmation
            long_entry = (bullish_cross and 
                         price_above_cloud and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: bearish TK cross + price below cloud + below 1d EMA + volume confirmation
            short_entry = (bearish_cross and 
                          price_below_cloud and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below cloud OR below 1d EMA OR bearish TK cross
            if (curr_close < cloud_bottom or 
                curr_close < ema_trend or 
                bearish_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above cloud OR above 1d EMA OR bullish TK cross
            if (curr_close > cloud_top or 
                curr_close > ema_trend or 
                bullish_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_DailyEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0