#!/usr/bin/env python3
# 12h_1w_1d_camarilla_pivot_volume_v1
# Hypothesis: 12h strategy using weekly and daily Camarilla pivot levels with volume confirmation.
# Long: Price touches weekly Camarilla H3 level with volume > 1.5x 20-period average and closes above H3.
# Short: Price touches weekly Camarilla L3 level with volume > 1.5x 20-period average and closes below L3.
# Exit: Opposite pivot touch or ATR trailing stop (2.0x ATR).
# Uses weekly Camarilla for major structure, daily for confirmation, volume for validity, ATR for dynamic stops.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for weekly Camarilla pivot (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    camarilla_h3_1w = pivot_1w + range_1w * 1.1 / 4
    camarilla_l3_1w = pivot_1w - range_1w * 1.1 / 4
    
    # Get 1d data for daily Camarilla pivot (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    camarilla_h3_1d = pivot_1d + range_1d * 1.1 / 4
    camarilla_l3_1d = pivot_1d - range_1d * 1.1 / 4
    
    # Align HTF Camarilla levels to 12h timeframe (wait for completed bar)
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_1w_aligned[i]) or np.isnan(camarilla_l3_1w_aligned[i]) or
            np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price touches daily Camarilla L3
            elif low[i] <= camarilla_l3_1d_aligned[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if short_low > 0 and close[i] > short_low + 2.0 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price touches daily Camarilla H3
            elif high[i] >= camarilla_h3_1d_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry: touch weekly Camarilla levels with volume confirmation and daily alignment
            weekly_h3_touch = (high[i] >= camarilla_h3_1w_aligned[i] and low[i] <= camarilla_h3_1w_aligned[i]) and volume_confirmed
            weekly_l3_touch = (high[i] >= camarilla_l3_1w_aligned[i] and low[i] <= camarilla_l3_1w_aligned[i]) and volume_confirmed
            
            # Additional filter: daily pivot alignment for confluence
            if weekly_h3_touch and close[i] > camarilla_h3_1d_aligned[i]:
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            elif weekly_l3_touch and close[i] < camarilla_l3_1d_aligned[i]:
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals