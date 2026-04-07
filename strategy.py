#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1d_trend_v1
Hypothesis: On 6-hour timeframe, use Ichimoku Cloud components from daily timeframe for trend direction and 6h TK Cross for entry timing.
Enter long when 6h Tenkan-sen crosses above Kijun-sen AND price is above daily Kumo (cloud).
Enter short when 6h Tenkan-sen crosses below Kijun-sen AND price is below daily Kumo (cloud).
Exit when TK Cross reverses or price crosses Kumo in opposite direction.
Ichimoku provides multi-layered trend support/resistance that works in both trending and ranging markets.
Daily timeframe filter ensures alignment with higher timeframe trend, reducing whipsaw.
Target: 20-40 trades/year to minimize fee drag while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Ichimoku on 6h timeframe for TK Cross
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Get daily data for Kumo (Cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Tenkan-sen and Kijun-sen for daily
    d_tenkan = (pd.Series(d_high).rolling(window=9, min_periods=9).max() + 
                pd.Series(d_low).rolling(window=9, min_periods=9).min()) / 2
    d_kijun = (pd.Series(d_high).rolling(window=26, min_periods=26).max() + 
               pd.Series(d_low).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    d_senkou_a = ((d_tenkan + d_kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(d_high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(d_low).rolling(window=52, min_periods=52).min()
    d_senkou_b = ((period52_high + period52_low) / 2)
    
    # Align daily components to 6h timeframe (shifted by 1 day to avoid look-ahead)
    d_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_a.values)
    d_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_b.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(26, n):  # Start after Kijun warmup
        # Skip if required data not available
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or
            np.isnan(d_senkou_a_aligned[i]) or np.isnan(d_senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # TK Cross signals
        tk_cross_above = (tenkan_sen.iloc[i] > kijun_sen.iloc[i]) and (tenkan_sen.iloc[i-1] <= kijun_sen.iloc[i-1])
        tk_cross_below = (tenkan_sen.iloc[i] < kijun_sen.iloc[i]) and (tenkan_sen.iloc[i-1] >= kijun_sen.iloc[i-1])
        
        # Kumo (Cloud) boundaries
        senkou_top = np.maximum(d_senkou_a_aligned[i], d_senkou_b_aligned[i])
        senkou_bottom = np.minimum(d_senkou_a_aligned[i], d_senkou_b_aligned[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > senkou_top
        price_below_cloud = close[i] < senkou_bottom
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when TK Cross turns bearish
            if tk_cross_below:
                exit_long = True
            # Exit when price drops below cloud
            elif price_below_cloud:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when TK Cross turns bullish
            if tk_cross_above:
                exit_short = True
            # Exit when price rises above cloud
            elif price_above_cloud:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TK Cross bullish AND price above cloud
            long_entry = tk_cross_above and price_above_cloud
            
            # Short entry: TK Cross bearish AND price below cloud
            short_entry = tk_cross_below and price_below_cloud
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals