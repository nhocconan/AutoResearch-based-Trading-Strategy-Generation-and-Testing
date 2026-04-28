#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Uses Ichimoku Cloud (Tenkan/Kijun/Senkou) from 1h data to identify trend direction and support/resistance.
# Enters long when price crosses above Kumo (cloud) in bullish trend (price > 1d EMA50),
# enters short when price crosses below Kumo in bearish trend (price < 1d EMA50).
# Volume confirmation (1.5x 20-period average) filters false breakouts.
# Designed for 6h timeframe with ~50-150 total trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Ichimoku calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1h data
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1h).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1h).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1h).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1h).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high_1h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low_1h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1h, senkou_span_b.values)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 50, 20)  # Wait for Ichimoku, EMA, and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Kumo (Cloud) boundaries
        upper_kumo = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_kumo = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Kumo breakout in trend direction with volume
        kumo_breakout_up = close[i] > upper_kumo and close[i-1] <= upper_kumo
        kumo_breakout_down = close[i] < lower_kumo and close[i-1] >= lower_kumo
        
        long_entry = kumo_breakout_up and uptrend and volume_confirm[i]
        short_entry = kumo_breakout_down and downtrend and volume_confirm[i]
        
        # Exit conditions: opposite Kumo breakout or loss of trend
        long_exit = kumo_breakout_down or (not uptrend)
        short_exit = kumo_breakout_up or (not downtrend)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0