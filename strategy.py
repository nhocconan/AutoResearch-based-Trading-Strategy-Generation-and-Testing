#!/usr/bin/env python3
# 1h_4h1d_camarilla_pivot_volume_spike_v1
# Hypothesis: 1h strategy using 4h trend and 1d Camarilla pivot structure for direction, with 1h volume spike for entry timing.
# Long: 4h close > 4h EMA20 (uptrend) AND 1h close breaks above H4 (1d Camarilla) with volume > 2.0x 20-period average.
# Short: 4h close < 4h EMA20 (downtrend) AND 1h close breaks below L4 (1d Camarilla) with volume > 2.0x 20-period average.
# Exit: Price returns to opposite Camarilla level (H3 for longs, L3 for shorts).
# Uses 1h primary timeframe with 4h HTF for trend and 1d HTF for Camarilla levels.
# Designed for low trade frequency (~15-37/year) to avoid fee drag on difficult 1h timeframe.
# Works in bull markets via 4h uptrend + breakouts and bear markets via 4h downtrend + fade-from-extremes logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_camarilla_pivot_volume_spike_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    ema20_4h = close_4h_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
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
    
    # Align 1d Camarilla levels to 1h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to H3
            if close[i] <= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3
            if close[i] >= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: 4h uptrend + Price breaks above H4 with volume and bullish candle
            if (close[i] > ema20_4h_aligned[i] and    # 4h uptrend
                close[i] > h4_1d_aligned[i] and       # Break above H4
                volume_confirmed and                  # Volume spike
                bullish_candle):                      # Bullish candle
                position = 1
                signals[i] = 0.20
            # Short entry: 4h downtrend + Price breaks below L4 with volume and bearish candle
            elif (close[i] < ema20_4h_aligned[i] and  # 4h downtrend
                  close[i] < l4_1d_aligned[i] and     # Break below L4
                  volume_confirmed and                # Volume spike
                  bearish_candle):                    # Bearish candle
                position = -1
                signals[i] = -0.20
    
    return signals