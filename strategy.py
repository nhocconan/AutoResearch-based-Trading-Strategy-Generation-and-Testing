#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h for Ichimoku calculations and entries/exits.
- HTF: 1d for trend direction (price above/below Kumo cloud) and 1w for major regime filter.
- Logic: Long when Tenkan > Kijun (bullish TK cross) AND price above Kumo (Senkou Span A/B) 
         AND 1d close > 1w EMA50 (primary uptrend) AND volume spike.
         Short when Tenkan < Kijun (bearish TK cross) AND price below Kumo 
         AND 1d close < 1w EMA50 (primary downtrend) AND volume spike.
- Exit: Opposite TK cross or loss of Kumo position.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for major regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align HTF indicators to 6h
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_b)
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 50)  # Need enough bars for Ichimoku (52) and 1w EMA (50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        close_1d_val = close_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        curr_close = close[i]
        
        # Kumo (Cloud) boundaries: Senkou Span A and B
        upper_kumo = max(senkou_a_val, senkou_b_val)
        lower_kumo = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Tenkan > Kijun (TK cross up) AND price above Kumo AND 1d close > 1w EMA50
                if (tenkan_val > kijun_val and 
                    curr_close > upper_kumo and 
                    close_1d_val > ema_50_1w_val):
                    signals[i] = 0.25
                    position = 1
                # Bearish: Tenkan < Kijun (TK cross down) AND price below Kumo AND 1d close < 1w EMA50
                elif (tenkan_val < kijun_val and 
                      curr_close < lower_kumo and 
                      close_1d_val < ema_50_1w_val):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Tenkan < Kijun (TK cross down) OR price falls below Kumo
            if (tenkan_val < kijun_val or 
                curr_close < lower_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan > Kijun (TK cross up) OR price rises above Kumo
            if (tenkan_val > kijun_val or 
                curr_close > upper_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTK_Cross_KumoFilter_1wEMA50Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0