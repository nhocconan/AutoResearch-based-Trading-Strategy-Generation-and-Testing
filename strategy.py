#!/usr/bin/env python3
# 6h_12h_1d_camarilla_pivot_volume_v1
# Hypothesis: 6h strategy using 12h and 1d Camarilla pivot levels with volume confirmation.
# Long: Price breaks above 1d H4 with volume > 1.8x 20-period average and 12h close > 12h open.
# Short: Price breaks below 1d L4 with volume > 1.8x 20-period average and 12h close < 12h open.
# Exit: Price returns to opposite Camarilla level (H3 for longs, L3 for shorts).
# Uses 12h trend filter: only long when 12h close > 12h EMA20, only short when 12h close < 12h EMA20.
# Target: 12-30 trades/year to minimize fee drag while maintaining edge.
# Works in bull markets via breakouts and bear markets via fade-from-extremes logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 6h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # 12h EMA20 for trend filter
    close_12h_s = pd.Series(close_12h)
    ema_20_12h = close_12h_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA20 and bullish/bearish candle to 6h
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    open_12h_aligned = align_htf_to_ltf(prices, df_12h, open_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(open_prices[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(open_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        # 12h bullish candle: close > open
        candle_12h_bullish = close_12h_aligned[i] > open_12h_aligned[i]
        # 12h bearish candle: close < open
        candle_12h_bearish = close_12h_aligned[i] < open_12h_aligned[i]
        # 12h trend filter: close > EMA20 for uptrend, < EMA20 for downtrend
        trend_12h_up = close_12h_aligned[i] > ema_20_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to H3
            if close[i] <= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3
            if close[i] >= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H4 with volume, 12h bullish candle, and uptrend
            if (close[i] > h4_1d_aligned[i] and    # Break above H4
                volume_confirmed and               # Volume spike
                candle_12h_bullish and             # 12h bullish candle
                trend_12h_up):                     # 12h uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 with volume, 12h bearish candle, and downtrend
            elif (close[i] < l4_1d_aligned[i] and  # Break below L4
                  volume_confirmed and             # Volume spike
                  candle_12h_bearish and           # 12h bearish candle
                  trend_12h_down):                 # 12h downtrend
                position = -1
                signals[i] = -0.25
    
    return signals