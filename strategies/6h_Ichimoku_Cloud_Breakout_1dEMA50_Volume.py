#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Ichimoku cloud provides dynamic support/resistance; breakouts above/below cloud with volume
# indicate strong momentum. 1d EMA50 ensures trades align with daily trend to avoid false signals.
# Works in bull markets (buying cloud breakouts in uptrend) and bear markets
# (selling cloud breakdowns in downtrend) by only taking trades in direction of 1d EMA50.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # Upper cloud boundary = max(Senkou A, Senkou B)
    # Lower cloud boundary = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: 2.0x 20-period average (~5.3 days for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Ichimoku calculation)
    start_idx = max(period_senkou_b + 26, 50)  # 52+26=78 for Senkou B, 50 for EMA
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku values are plotted ahead/behind, so we use current values
        # Tenkan and Kijun are for current period
        # Senkou spans are plotted 26 periods ahead, so we use values from 26 periods ago
        # For current cloud, we use Senkou A/B from 26 periods ago
        idx_cloud = i - 26
        if idx_cloud < 0:
            signals[i] = 0.0
            continue
            
        upper_cloud_current = upper_cloud[idx_cloud]
        lower_cloud_current = lower_cloud[idx_cloud]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper cloud with volume spike AND price > 1d EMA50 (bullish trend)
            if (close[i] > upper_cloud_current and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower cloud with volume spike AND price < 1d EMA50 (bearish trend)
            elif (close[i] < lower_cloud_current and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below lower cloud (cloud break) OR price below 1d EMA50 (trend change)
            if close[i] < lower_cloud_current or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above upper cloud (cloud break) OR price above 1d EMA50 (trend change)
            if close[i] > upper_cloud_current or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals