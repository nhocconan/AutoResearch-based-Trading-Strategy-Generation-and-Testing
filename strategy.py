#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_ichimoku_trend_v1
# Ichimoku Cloud system on 1d timeframe with 6h entry timing.
# Uses Tenkan/Kijun cross and price relative to Kumo (cloud) for trend direction.
# In bull markets: price above cloud + TK cross up = long.
# In bear markets: price below cloud + TK cross down = short.
# Volume confirmation filters weak signals. Target: 15-30 trades/year per symbol.
name = "6h_1d_ichimoku_trend_v1"
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (standard periods: 9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period_kijun)  # shifted 26 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(period_kijun)
    
    # Chikou Span (Lagging Span): close shifted back 26 periods
    chikou_span = close_1d.shift(-period_kijun)  # Note: negative shift for lagging
    
    # Align all components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span.values)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after warmup for Senkou Span B
        # Skip if any Ichimoku component not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_kumo = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Check TK cross conditions
        tk_cross_up = (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                       tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        tk_cross_down = (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                         tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        # Price relative to cloud
        price_above_kumo = close[i] > upper_kumo
        price_below_kumo = close[i] < lower_kumo
        
        # Long signal: price above cloud + TK cross up
        if price_above_kumo and tk_cross_up and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price below cloud + TK cross down
        elif price_below_kumo and tk_cross_down and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite TK cross or price crosses opposite cloud edge
        elif ((tk_cross_down and position == 1) or 
              (price_below_kumo and position == 1) or
              (tk_cross_up and position == -1) or
              (price_above_kumo and position == -1)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals