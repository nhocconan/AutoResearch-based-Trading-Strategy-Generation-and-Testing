#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_reversal_volume_v2
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and reversal entry.
# Long: Price touches or crosses below Camarilla L3 level on 1d, then reverses upward with volume > 1.5x 20-period average on 12h.
# Short: Price touches or crosses above Camarilla H3 level on 1d, then reverses downward with volume > 1.5x 20-period average on 12h.
# Exit: Opposite Camarilla level touch or ATR trailing stop (2.5x ATR from extreme).
# Uses daily Camarilla for structure, 12h price action for reversal entry, volume for confirmation, ATR for dynamic stops.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_reversal_volume_v2"
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
    open_price = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3 = pivot + (range_1d * 1.1 / 4.0)
    l3 = pivot - (range_1d * 1.1 / 4.0)
    h4 = pivot + (range_1d * 1.1 / 2.0)
    l4 = pivot - (range_1d * 1.1 / 2.0)
    
    # Align HTF Camarilla levels to 12h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    long_triggered = False  # flag to wait for reversal after L3 touch
    short_triggered = False  # flag to wait for reversal after H3 touch
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(open_price[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                long_triggered = False
                signals[i] = 0.0
            # Exit: Price touches or crosses above H3 level
            elif high[i] >= h3_aligned[i]:
                position = 0
                long_high = 0.0
                long_triggered = False
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if short_low > 0 and close[i] > short_low + 2.5 * atr[i]:
                position = 0
                short_low = 0.0
                short_triggered = False
                signals[i] = 0.0
            # Exit: Price touches or crosses below L3 level
            elif low[i] <= l3_aligned[i]:
                position = 0
                short_low = 0.0
                short_triggered = False
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for L3 touch (long setup) or H3 touch (short setup)
            long_setup = (low[i] <= l3_aligned[i]) and volume_confirmed
            short_setup = (high[i] >= h3_aligned[i]) and volume_confirmed
            
            if long_setup:
                long_triggered = True
                short_triggered = False
            elif short_setup:
                short_triggered = True
                long_triggered = False
            
            # Long entry: after L3 touch, price reverses upward (close above open)
            if long_triggered and close[i] > open_price[i]:
                position = 1
                long_high = high[i]
                long_triggered = False
                signals[i] = 0.25
            # Short entry: after H3 touch, price reverses downward (close below open)
            elif short_triggered and close[i] < open_price[i]:
                position = -1
                short_low = low[i]
                short_triggered = False
                signals[i] = -0.25
    
    return signals