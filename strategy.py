#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and ATR stoploss.
# Long: Price touches Camarilla H3 level from below with volume > 1.5x 20-period average, ATR trailing stop.
# Short: Price touches Camarilla L3 level from above with volume > 1.5x 20-period average, ATR trailing stop.
# Exit: ATR trailing stop (2.0x ATR from extreme) or opposite Camarilla level touch.
# Uses daily Camarilla for key support/resistance, 12h for execution, volume for confirmation, ATR for dynamic stops.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
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
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3 = pivot + (range_1d * 1.1 / 2)
    l3 = pivot - (range_1d * 1.1 / 2)
    h4 = pivot + (range_1d * 1.1)
    l4 = pivot - (range_1d * 1.1)
    
    # Align HTF Camarilla levels to 12h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
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
            # Exit: Price touches or goes above Camarilla H4 level
            elif close[i] >= h4_aligned[i]:
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
            # Exit: Price touches or goes below Camarilla L4 level
            elif close[i] <= l4_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Camarilla level touch with volume confirmation
            # Long: Price touches H3 from below (close crosses above H3)
            bullish_touch = (close[i] > h3_aligned[i]) and (close[i-1] <= h3_aligned[i-1] if i > 0 else False) and volume_confirmed
            # Short: Price touches L3 from above (close crosses below L3)
            bearish_touch = (close[i] < l3_aligned[i]) and (close[i-1] >= l3_aligned[i-1] if i > 0 else False) and volume_confirmed
            
            if bullish_touch:
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            elif bearish_touch:
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals