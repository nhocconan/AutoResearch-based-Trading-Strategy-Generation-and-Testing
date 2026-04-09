#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_reversal_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels for mean reversion in ranging markets.
# Long: Price touches L3 with volume < 0.8x 20-period average (low volume rejection) and closes above open.
# Short: Price touches H3 with volume < 0.8x 20-period average and closes below open.
# Exit: Price reaches opposite Camarilla level (H4 for longs, L4 for shorts) or midpoint (pivot).
# Uses 1w EMA filter: only trade long when price > weekly EMA50, short when price < weekly EMA50.
# Target: 12-30 trades/year to minimize fee drag while capturing range reversals.
# Works in bull markets via fade-from-H3 and in bear markets via bounce-from-L3.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_reversal_v1"
timeframe = "12h"
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
    
    # Align 1d Camarilla levels to 12h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(open_prices[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volume filter: current volume < 0.8x 20-period average (rejection signal)
        low_volume = volume[i] < 0.8 * volume_ma[i]
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price reaches H4 (profit target) or pivot (partial exit)
            if close[i] >= h4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] <= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches L4 (profit target) or pivot (partial exit)
            if close[i] <= l4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] >= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price touches L3 with low volume rejection and bullish candle, above weekly EMA
            if (abs(close[i] - l3_1d_aligned[i]) < 0.001 * close[i] and  # Touches L3 (within 0.1%)
                low_volume and                                          # Low volume rejection
                bullish_candle and                                      # Bullish candle
                close[i] > ema_50_1w_aligned[i]):                       # Above weekly EMA (long bias)
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches H3 with low volume rejection and bearish candle, below weekly EMA
            elif (abs(close[i] - h3_1d_aligned[i]) < 0.001 * close[i] and  # Touches H3 (within 0.1%)
                  low_volume and                                           # Low volume rejection
                  bearish_candle and                                       # Bearish candle
                  close[i] < ema_50_1w_aligned[i]):                        # Below weekly EMA (short bias)
                position = -1
                signals[i] = -0.25
    
    return signals