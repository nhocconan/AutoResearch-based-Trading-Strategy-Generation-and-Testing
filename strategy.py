#!/usr/bin/env python3
"""
12h_1w_Camarilla_Pivot_Breakout_WeeklyTrend_v1
Hypothesis: Use weekly CCI trend filter with weekly Camarilla breakout and volume confirmation on 12h.
Long when price breaks above weekly H4 with volume > 2x 20-period average AND weekly CCI > -100 (uptrend).
Short when price breaks below weekly L4 with volume > 2x 20-period average AND weekly CCI < 100 (downtrend).
Designed for 12h timeframe to target 12-37 trades/year with high win rate in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (vol_ma_20 * 2.0)
    
    # Get weekly data for Camarilla levels and CCI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (H4 and L4)
    camarilla_h4_1w = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 2
    camarilla_l4_1w = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 2
    
    # Calculate weekly CCI (20-period)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    tp_ma = typical_price.rolling(window=20, min_periods=20).mean()
    tp_md = typical_price.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    cci = (typical_price - tp_ma) / (0.015 * tp_md)
    cci_values = cci.values
    
    # Align weekly levels and CCI to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci_values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(40, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(cci_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above weekly H4 with volume filter AND weekly CCI > -100 (not strong downtrend)
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_filter[i] and 
                      cci_aligned[i] > -100)
        
        # Short signal: break below weekly L4 with volume filter AND weekly CCI < 100 (not strong uptrend)
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_filter[i] and 
                       cci_aligned[i] < 100)
        
        if position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price breaks below L4 with volume confirmation
            if (close[i] < camarilla_l4_aligned[i] and volume_filter[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when price breaks above H4 with volume confirmation
            if (close[i] > camarilla_h4_aligned[i] and volume_filter[i]):
                position = 1
                signals[i] = position_size
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Camarilla_Pivot_Breakout_WeeklyTrend_v1"
timeframe = "12h"
leverage = 1.0