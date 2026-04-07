#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v8
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from daily timeframe with volume confirmation.
Long when price touches or breaks below Camarilla L3 with bullish reversal candle and volume > 1.5x average.
Short when price touches or breaks above Camarilla H3 with bearish reversal candle and volume > 1.5x average.
Exit when price reaches Camarilla H4 (for longs) or L4 (for shorts).
Designed for 15-30 trades/year to minimize fee drag while capturing mean reversions in range-bound markets.
Works in both bull/bear markets as Camarilla levels adapt to volatility and volume filter avoids false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v8"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_h4 = close_1d + range_1d * 1.1 / 2
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_h2 = close_1d + range_1d * 1.1 / 6
    camarilla_h1 = close_1d + range_1d * 1.1 / 12
    camarilla_l1 = close_1d - range_1d * 1.1 / 12
    camarilla_l2 = close_1d - range_1d * 1.1 / 6
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    camarilla_l4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume filter: 20-period average on 12h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Camarilla H4
            if close[i] >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla L4
            if close[i] <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Bullish reversal: close > open
                bullish = close[i] > prices['open'].iloc[i]
                # Bearish reversal: close < open
                bearish = close[i] < prices['open'].iloc[i]
                
                # Long: price touches/below L3 with bullish reversal
                if close[i] <= camarilla_l3_aligned[i] and bullish:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches/above H3 with bearish reversal
                elif close[i] >= camarilla_h3_aligned[i] and bearish:
                    position = -1
                    signals[i] = -0.25
    
    return signals