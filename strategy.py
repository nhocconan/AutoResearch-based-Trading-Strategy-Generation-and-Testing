#!/usr/bin/env python3
"""
1h_volume_breakout_mtf_v1
Hypothesis: On 1h timeframe, enter long when price breaks above the 10-period high with volume > 1.5x average volume and 4h close > 1d close (bullish regime). Enter short when price breaks below the 10-period low with volume > 1.5x average volume and 4h close < 1d close (bearish regime). Exit when price breaks in opposite direction or volume dries up. Uses 4h/1d for trend regime filter and 1h for entry timing to reduce false signals. Target: 15-35 trades/year per symbol to minimize fee drag while capturing momentum bursts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_breakout_mtf_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 10-period high/low for breakout
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h and 1d data for regime filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h and 1d close prices for regime
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Align HTF closes to 1h timeframe (with shift(1) for completed bars)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Session filter: 8-20 UTC (already datetime64 index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_10[i]) or np.isnan(low_10[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close_4h_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: >1.5x average volume
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 10-period low OR volume dries up
            if low[i] < low_10[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above 10-period high OR volume dries up
            if high[i] > high_10[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish regime: 4h close > 1d close (bullish longer-term trend)
                bullish_regime = close_4h_aligned[i] > close_1d_aligned[i]
                # Bearish regime: 4h close < 1d close (bearish longer-term trend)
                bearish_regime = close_4h_aligned[i] < close_1d_aligned[i]
                
                # Long: break above 10-period high in bullish regime
                if high[i] > high_10[i] and bullish_regime:
                    position = 1
                    signals[i] = 0.20
                # Short: break below 10-period low in bearish regime
                elif low[i] < low_10[i] and bearish_regime:
                    position = -1
                    signals[i] = -0.20
    
    return signals