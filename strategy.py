#!/usr/bin/env python3
name = "12h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Ichimoku Cloud components (9, 26, 52)
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (pd.Series(high).rolling(window=52, min_periods=52).max().values + 
                     pd.Series(low).rolling(window=52, min_periods=52).min().values) / 2
    
    # Align Ichimoku components to 12h timeframe
    tenkan_12h = align_htf_to_ltf(prices, prices, tenkan_sen)
    kijun_12h = align_htf_to_ltf(prices, prices, kijun_sen)
    senkou_a_12h = align_htf_to_ltf(prices, prices, senkou_span_a)
    senkou_b_12h = align_htf_to_ltf(prices, prices, senkou_span_b)
    
    # Daily EMA26 for trend filter
    close_1d = df_1d['close'].values
    ema_26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_12h = align_htf_to_ltf(prices, df_1d, ema_26_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_12h[i]) or np.isnan(kijun_12h[i]) or 
            np.isnan(senkou_a_12h[i]) or np.isnan(senkou_b_12h[i]) or 
            np.isnan(ema_26_12h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_12h[i], senkou_b_12h[i])
        cloud_bottom = min(senkou_a_12h[i], senkou_b_12h[i])
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above cloud in daily uptrend with volume
            if close[i] > cloud_top and ema_26_12h[i] > ema_26_12h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud in daily downtrend with volume
            elif close[i] < cloud_bottom and ema_26_12h[i] < ema_26_12h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to cloud or trend reverses
            if close[i] < cloud_top or ema_26_12h[i] < ema_26_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to cloud or trend reverses
            if close[i] > cloud_bottom or ema_26_12h[i] > ema_26_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Cloud breakouts with daily trend filter and volume confirmation
# - Ichimoku Cloud provides dynamic support/resistance; breakouts indicate strong momentum
# - Daily EMA26 trend filter ensures trades align with higher-timeframe direction
# - Volume confirmation (2x average) reduces false breakouts
# - Works in both bull (cloud breakouts in uptrend) and bear (cloud breakdowns in downtrend)
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Uses 12h timeframe for execution, 1d for trend and volume context
# - Ichimoku is underutilized in crypto; offers unique edge vs. saturated strategies
# - Focus on BTC/ETH as primary targets with proven cloud effectiveness in trending markets