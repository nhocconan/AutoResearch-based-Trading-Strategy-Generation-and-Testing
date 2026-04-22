#!/usr/bin/env python3

"""
Hypothesis: 6-hour Ichimoku Cloud strategy with 1-day trend filter and volume confirmation.
Uses Ichimoku conversion line (9-period), base line (26-period), leading span A (26-period), and leading span B (52-period) from 6h data.
Trades when price is above/below the cloud with TK cross confirmation, aligned with 1-day EMA(34) trend.
Volume filter requires current volume > 1.5x 20-period average to avoid low-conviction moves.
Targets 50-150 total trades over 4 years (12-37/year) with disciplined entries to minimize fee drag.
Works in bull markets via cloud breakouts and in bear markets via trend-aligned mean reversion at cloud edges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Ichimoku calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku Components (6h)
    # Conversion Line (Tenkan-sen): (9-period high + low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h lower timeframe (no shift needed as Ichimoku is forward-looking)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Cloud boundaries (use current values, not shifted)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0 and vol_spike:
            # Long: price above cloud + TK cross bullish (Tenkan > Kijun)
            if (close[i] > upper_cloud and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross bearish (Tenkan < Kijun)
            elif (close[i] < lower_cloud and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price enters cloud or TK cross reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price drops below cloud or TK cross turns bearish
                if (close[i] < upper_cloud or 
                    tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above cloud or TK cross turns bullish
                if (close[i] > lower_cloud or 
                    tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0