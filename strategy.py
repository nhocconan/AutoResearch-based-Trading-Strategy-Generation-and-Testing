#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1-day Camarilla H4/L4 breakout with volume and 4h EMA200 trend filter.
# Camarilla H4/L4 are stronger reversal levels (breakouts indicate trend continuation).
# Only trade breakouts aligned with 4h EMA200 trend to avoid false signals.
# Volume surge confirms institutional participation.
# Designed for low-frequency, high-conviction trades (~20-30/year) to minimize fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns with trend).
name = "4h_Camarilla_H4L4_EMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Camarilla levels (based on prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla H4/L4 = close ± (high-low)*1.1
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1
    
    # Align to 4h timeframe (waits for prior day close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # EMA200 on 4h close (trend filter)
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume spike: volume > 2.0 * 50-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema200[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above H4 with uptrend and volume
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] > ema200[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below L4 with downtrend and volume
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] < ema200[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches L4 or breaks below EMA200
            if (close[i] < camarilla_l4_aligned[i]) or (close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches H4 or breaks above EMA200
            if (close[i] > camarilla_h4_aligned[i]) or (close[i] > ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals