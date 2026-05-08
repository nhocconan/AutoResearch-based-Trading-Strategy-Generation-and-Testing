#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku cloud and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    high_9 = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    
    tenkan_sen = (high_9 + low_9) / 2
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get weekly data for trend filter (20-period EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for volume spike (20-period average)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current day's data (last completed day)
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed day
        
        if idx_1d < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_current = df_1d['volume'].iloc[idx_1d]
        vol_avg_20_current = vol_avg_20[idx_1d]
        
        if np.isnan(vol_current) or np.isnan(vol_avg_20_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 2.0x 20-period average
        vol_confirmed = vol_current > 2.0 * vol_avg_20_current
        
        # Current price
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_span_a_aligned[i]
        senkou_b = senkou_span_b_aligned[i]
        ema20_1w = ema20_1w_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                # Long when Tenkan crosses above Kijun and price is above cloud and above weekly EMA20
                if tenkan > kijun and price > cloud_top and price > ema20_1w:
                    signals[i] = 0.25
                    position = 1
                # Short when Tenkan crosses below Kijun and price is below cloud and below weekly EMA20
                elif tenkan < kijun and price < cloud_bottom and price < ema20_1w:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Manage long position
            exit_signal = False
            # Exit when price crosses below cloud or Tenkan crosses below Kijun
            if price < cloud_bottom or tenkan < kijun:
                exit_signal = True
            # Exit when volume confirmation lost
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            # Exit when price crosses above cloud or Tenkan crosses above Kijun
            if price > cloud_top or tenkan > kijun:
                exit_signal = True
            # Exit when volume confirmation lost
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals