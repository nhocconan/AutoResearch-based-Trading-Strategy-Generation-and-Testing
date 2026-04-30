#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku cloud breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Kumo (cloud), Tenkan > Kijun, close > 12h EMA50, and volume > 2.0x 20-bar avg.
# Short when price breaks below Kumo (cloud), Tenkan < Kijun, close < 12h EMA50, and volume > 2.0x 20-bar avg.
# Exit when price re-enters the Kumo (cloud).
# Uses Ichimoku from 6h timeframe for structure, 12h EMA50 for higher timeframe trend filter.
# Volume confirmation reduces false breakouts. Discrete position sizing at ±0.25 to balance performance and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Ichimoku_Kumo_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Kumo (cloud) boundaries: Senkou Span A and B shifted forward by 26 periods
    # For signal at time t, we use Senkou Span A/B from t-26 (already calculated)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Upper cloud boundary: max(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_shifted, senkou_b_shifted)
    # Lower cloud boundary: min(Senkou A, Senkou B)
    lower_cloud = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # warmup for Ichimoku (52+26) and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_upper_cloud = upper_cloud[i]
        curr_lower_cloud = lower_cloud[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper cloud, Tenkan > Kijun, close > 12h EMA50, volume spike
            if (curr_close > curr_upper_cloud and 
                curr_tenkan > curr_kijun and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower cloud, Tenkan < Kijun, close < 12h EMA50, volume spike
            elif (curr_close < curr_lower_cloud and 
                  curr_tenkan < curr_kijun and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters the cloud (below upper cloud)
            if curr_close < curr_upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters the cloud (above lower cloud)
            if curr_close > curr_lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals