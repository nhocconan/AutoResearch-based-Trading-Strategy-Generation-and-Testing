#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 20-40/year) with clear mean reversion logic
# Works in both bull (pullback to support) and bear (bounce from resistance) markets
# Uses volume spike and trend filter to avoid false reversals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Camarilla levels (based on previous day's range)
    # Calculate daily range from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 4h bar based on previous day's range
    camarilla_h5 = np.zeros(len(close_4h))
    camarilla_h4 = np.zeros(len(close_4h))
    camarilla_h3 = np.zeros(len(close_4h))
    camarilla_l3 = np.zeros(len(close_4h))
    camarilla_l4 = np.zeros(len(close_4h))
    camarilla_l5 = np.zeros(len(close_4h))
    
    # For each 4h bar, use previous day's range
    for i in range(len(close_4h)):
        # Find corresponding 1d index (previous day)
        # Since we don't have direct mapping, use approximation: 16 4h bars per day
        idx_1d = max(0, i // 16 - 1)  # Previous day
        if idx_1d < len(high_1d):
            day_high = high_1d[idx_1d]
            day_low = low_1d[idx_1d]
            day_close = close_1d[idx_1d]
            range_val = day_high - day_low
            
            camarilla_h5[i] = day_close + 1.1 * range_val * 1.1
            camarilla_h4[i] = day_close + 1.1 * range_val * 0.55
            camarilla_h3[i] = day_close + 1.1 * range_val * 0.275
            camarilla_l3[i] = day_close - 1.1 * range_val * 0.275
            camarilla_l4[i] = day_close - 1.1 * range_val * 0.55
            camarilla_l5[i] = day_close - 1.1 * range_val * 1.1
    
    # Load 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for volatility and stoploss (14-period on 4h)
    tr1 = np.maximum(high_4h[1:], low_4h[:-1]) - np.minimum(high_4h[1:], low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l5)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)
        
        # Long entry: price touches/bounces from L3 level + uptrend + volume spike
        if (close[i] <= camarilla_l3_aligned[i] * 1.002 and  # Allow small buffer
            close[i] > camarilla_l4_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 1.8 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price touches/rejects from H3 level + downtrend + volume spike
        elif (close[i] >= camarilla_h3_aligned[i] * 0.998 and  # Allow small buffer
              close[i] < camarilla_h4_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 1.8 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or mean reversion to midpoint
        elif position == 1 and (close[i] >= camarilla_h3_aligned[i] or 
                                close[i] <= camarilla_l3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= camarilla_l3_aligned[i] or 
                                 close[i] >= camarilla_h3_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_1dVolume_1wEMA_Reversal"
timeframe = "4h"
leverage = 1.0