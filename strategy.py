#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when price > cloud, Tenkan > Kijun, and price > 1d EMA50 + volume spike
# Short when price < cloud, Tenkan < Kijun, and price < 1d EMA50 + volume spike
# Uses Ichimoku from 6h for structure, 1d EMA for higher timeframe trend, volume for confirmation
# Designed for low trade frequency (12-37/year on 6h) to minimize fee drag
# Works in bull (price above cloud with trend) and bear (price below cloud with trend) markets

name = "6h_Ichimoku_Cloud_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # We need current cloud values (Senkou A/B from 26 periods ago)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # Fill the first 26 values with NaN (no cloud data available)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Upper cloud boundary = max(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_lagged, senkou_b_lagged)
    # Lower cloud boundary = min(Senkou A, Senkou B)
    lower_cloud = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(52, 50, 20) + 26  # Ichimoku(52), EMA(50), volume MA(20), cloud shift(26)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > upper cloud, Tenkan > Kijun, price > 1d EMA50, volume spike
            if (close[i] > upper_cloud[i] and tenkan[i] > kijun[i] and 
                close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < lower cloud, Tenkan < Kijun, price < 1d EMA50, volume spike
            elif (close[i] < lower_cloud[i] and tenkan[i] < kijun[i] and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < lower cloud or Tenkan < Kijun or price < 1d EMA50
            if (close[i] < lower_cloud[i] or tenkan[i] < kijun[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > upper cloud or Tenkan > Kijun or price > 1d EMA50
            if (close[i] > upper_cloud[i] or tenkan[i] > kijun[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals