#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wCloud_Filter
Hypothesis: 6h Ichimoku TK cross (Tenkan/Kijun) with weekly cloud filter and volume confirmation.
In bull markets: price above cloud + TK cross up = long. In bear markets: price below cloud + TK cross down = short.
Weekly cloud acts as major trend filter to avoid whipsaws. Volume confirmation ensures momentum.
Designed for 12-30 trades/year on BTC/ETH with controlled risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for Ichimoku cloud (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # need at least 52 weeks for Ichimoku
        return np.zeros(n)
    
    # Ichimoku calculations on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(df_1w['close']).shift(26).values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1w, chikou_span)
    
    # 6h ATR for dynamic filtering
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations
    start_idx = 52  # need 52 weeks of data for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        top_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        bottom_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Ichimoku signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = curr_close > top_cloud
        price_below_cloud = curr_close < bottom_cloud
        price_in_cloud = (curr_close >= bottom_cloud) & (curr_close <= top_cloud)
        
        # Chikou span confirmation (lagging span should confirm price action)
        # For long: Chikou should be above price from 26 periods ago
        # For short: Chikou should be below price from 26 periods ago
        chikou_confirms_long = chikou_aligned[i] > curr_close
        chikou_confirms_short = chikou_aligned[i] < curr_close
        
        if position == 0:
            # Look for entry signals
            # Long: TK cross up + price above cloud + volume spike + Chikou confirms
            long_signal = (tk_cross_up and price_above_cloud and volume_spike[i] and 
                          chikou_confirms_long)
            # Short: TK cross down + price below cloud + volume spike + Chikou confirms
            short_signal = (tk_cross_down and price_below_cloud and volume_spike[i] and 
                           chikou_confirms_short)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price falls below cloud (trend change)
            # Exit if TK cross down (momentum loss)
            # Exit if Chikou no longer confirms
            if (price_below_cloud or not tk_cross_up or not chikou_confirms_long):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price rises above cloud (trend change)
            # Exit if TK cross up (momentum loss)
            # Exit if Chikou no longer confirms
            if (price_above_cloud or not tk_cross_down or not chikou_confirms_short):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wCloud_Filter"
timeframe = "6h"
leverage = 1.0