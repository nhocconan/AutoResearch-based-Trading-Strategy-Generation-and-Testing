#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud strategy with 1d trend filter and volume confirmation
# - Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h data
# - Long when price > cloud AND Tenkan > Kijun AND 1d close > 1d EMA50
# - Short when price < cloud AND Tenkan < Kijun AND 1d close < 1d EMA50
# - Volume filter: current 6h volume > 1.5x 20-period average of 1d volume (scaled)
# - Designed to capture strong trends while avoiding whipsaws in ranging markets
# - Target: 15-30 trades/year to minimize fee drag

name = "6h_Ichimoku_1dTrend_Volume_v1"
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
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # We'll use the current cloud values (not shifted) for simplicity
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough data for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 6h: 1d has 4x 6h bars, so divide by 4
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 4.0)
        
        # Ichimoku signals
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        if position == 0:
            # Long entry: price above cloud + bullish TK cross + uptrend on 1d + volume
            if (price_above_cloud and tenkan_above_kijun and 
                close[i] > ema_50_1d_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud + bearish TK cross + downtrend on 1d + volume
            elif (price_below_cloud and tenkan_below_kijun and 
                  close[i] < ema_50_1d_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below cloud or bearish TK cross
            if price_below_cloud or tenkan_below_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above cloud or bullish TK cross
            if price_above_cloud or tenkan_above_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals