#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when price > Kumo (cloud), Tenkan > Kijun, and price > 1d EMA200
# Short when price < Kumo (cloud), Tenkan < Kijun, and price < 1d EMA200
# Exit when price re-enters Kumo or Tenkan/Kijun cross reverses
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to avoid overtrading.
# Ichimoku provides dynamic support/resistance via Kumo and momentum via TK cross.
# Works in bull markets by capturing uptrends above cloud and in bear markets by shorting downtrends below cloud.
# The 1d EMA200 filter ensures alignment with higher timeframe trend, reducing false signals.

name = "6h_Ichimoku_Cloud_TK_Cross_1dEMA200_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Current Kumo (cloud) boundaries: Senkou Span A and B
    # Note: In real-time, the cloud is plotted 26 periods ahead, so we use current Senkou spans
    # For simplicity, we use current Senkou A/B as cloud boundaries (standard approach)
    # The cloud is between senkou_a and senkou_b
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 200)  # Ichimoku needs 52 periods, EMA200 needs 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a[i]
        curr_senkou_b = senkou_b[i]
        curr_ema200_1d = ema_200_1d_aligned[i]
        curr_close = close[i]
        
        # Kumo (cloud) boundaries
        upper_cloud = max(curr_senkou_a, curr_senkou_b)
        lower_cloud = min(curr_senkou_a, curr_senkou_b)
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price re-enters Kumo OR Tenkan crosses below Kijun
            if curr_close < upper_cloud or curr_tenkan < curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Kumo OR Tenkan crosses above Kijun
            if curr_close > lower_cloud or curr_tenkan > curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price > Kumo (cloud), Tenkan > Kijun, and price > 1d EMA200
            if curr_close > upper_cloud and curr_tenkan > curr_kijun and curr_close > curr_ema200_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price < Kumo (cloud), Tenkan < Kijun, and price < 1d EMA200
            elif curr_close < lower_cloud and curr_tenkan < curr_kijun and curr_close < curr_ema200_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals