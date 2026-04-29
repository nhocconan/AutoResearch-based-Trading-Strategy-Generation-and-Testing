#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1w trend filter and volume confirmation
# Long when Tenkan > Kijun AND price > Kumo cloud AND price > 1w EMA50 AND volume > 1.5x 20-period average
# Short when Tenkan < Kijun AND price < Kumo cloud AND price < 1w EMA50 AND volume > 1.5x 20-period average
# Uses price > Kumo cloud as trend filter to avoid whipsaw in ranging markets
# Weekly EMA50 ensures alignment with major trend, reducing counter-trend trades
# Volume confirmation ensures Ichimoku signals have strong participation
# Ichimoku works in bull markets via bullish TK cross above cloud with weekly uptrend
# Works in bear markets via bearish TK cross below cloud with weekly downtrend
# Target: 12-37 trades/year on 6h timeframe to minimize fee drag while capturing strong trends

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ichimoku components (using daily data for proper Ichimoku calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 52)  # warmup for Ichimoku (needs 52 periods)
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        
        # Skip if Ichimoku components are not available
        if (np.isnan(curr_tenkan) or np.isnan(curr_kijun) or 
            np.isnan(curr_senkou_a) or np.isnan(curr_senkou_b)):
            signals[i] = 0.0
            continue
        
        # Determine Kumo cloud boundaries (Senkou Span A and B)
        upper_cloud = max(curr_senkou_a, curr_senkou_b)
        lower_cloud = min(curr_senkou_a, curr_senkou_b)
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits (exit when Ichimoku signal reverses)
        if position == 1:  # Long position
            # Exit conditions: bearish TK cross OR price below cloud
            if curr_tenkan < curr_kijun or curr_close < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: bullish TK cross OR price above cloud
            if curr_tenkan > curr_kijun or curr_close > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish TK cross AND price above cloud AND price > 1w EMA50 AND volume spike
            if (curr_tenkan > curr_kijun and 
                curr_close > upper_cloud and 
                curr_close > curr_ema_1w and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TK cross AND price below cloud AND price < 1w EMA50 AND volume spike
            elif (curr_tenkan < curr_kijun and 
                  curr_close < lower_cloud and 
                  curr_close < curr_ema_1w and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals