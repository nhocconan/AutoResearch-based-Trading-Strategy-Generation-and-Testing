#!/usr/bin/env python3
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
    
    # === Daily data for Ichimoku components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted forward 26 periods
    highest_senkou_b = pd.Series(high_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).max().values
    lowest_senkou_b = pd.Series(low_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).min().values
    senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted back 26 periods (not used for entry)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6x ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_6 = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Volume spike detection (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(atr_6[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_6h[i]
        kijun = kijun_sen_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        atr_val = atr_6[i]
        vol_spike = volume_spike[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below cloud or Tenkan-Kijun cross down
            if price < cloud_bottom or (i > 0 and tenkan < kijun and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above cloud or Tenkan-Kijun cross up
            if price > cloud_top or (i > 0 and tenkan > kijun and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above cloud, Tenkan crosses above Kijun, with volume spike and ATR filter
            if (price > cloud_top and 
                tenkan > kijun and 
                tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1] and  # crossed this bar
                vol_spike and 
                atr_val > np.nanmedian(atr_6[max(0, i-50):i+1]) * 0.8):  # volatility not too low
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below cloud, Tenkan crosses below Kijun, with volume spike and ATR filter
            elif (price < cloud_bottom and 
                  tenkan < kijun and 
                  tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1] and  # crossed this bar
                  vol_spike and 
                  atr_val > np.nanmedian(atr_6[max(0, i-50):i+1]) * 0.8):  # volatility not too low
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_Volume"
timeframe = "6h"
leverage = 1.0