#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_1dFilter
Hypothesis: On 6h timeframe, Ichimoku cloud (Tenkan/Kijun/Senkou) from 1d HTF provides strong trend direction and dynamic support/resistance.
Price above/below cloud determines trend, TK cross provides entry signals, and cloud acts as trailing stop.
Uses 1d EMA50 as additional trend filter to avoid whipsaw in ranging markets.
Designed for 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
Works in bull markets via trend continuation and bear markets via trend reversals at cloud edges.
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
    
    # 1d data for Ichimoku and EMA50 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (1d)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=period_tenkan, min_periods=period_tenkan).max() +
                  pd.Series(df_1d['low']).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=period_kijun, min_periods=period_kijun).max() +
                 pd.Series(df_1d['low']).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(df_1d['high']).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() +
                      pd.Series(df_1d['low']).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) + EMA (50) + volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Cloud top and bottom (Senkou Span A/B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Trend filter: price must be on correct side of cloud AND EMA50
        bullish_trend = (curr_close > cloud_top) and (curr_close > ema_50_aligned[i])
        bearish_trend = (curr_close < cloud_bottom) and (curr_close < ema_50_aligned[i])
        
        # TK Cross signals
        tk_cross_bull = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_bear = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: TK cross in trend direction + volume spike
            long_entry = bullish_trend and tk_cross_bull and volume_spike[i]
            short_entry = bearish_trend and tk_cross_bear and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below cloud bottom (trend invalidation) or TK cross bear
            if curr_close < cloud_bottom or (tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above cloud top (trend invalidation) or TK cross bull
            if curr_close > cloud_top or (tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_1dFilter"
timeframe = "6h"
leverage = 1.0