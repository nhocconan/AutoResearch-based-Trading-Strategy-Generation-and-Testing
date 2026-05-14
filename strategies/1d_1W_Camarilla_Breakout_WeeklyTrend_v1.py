#!/usr/bin/env python3
"""
1d_1W_Camarilla_Breakout_WeeklyTrend_v1
Hypothesis: Use weekly CCI trend filter with daily Camarilla breakout and volume confirmation.
Long when price breaks above daily H4 with volume > 2x 20-day avg AND weekly CCI > -100 (uptrend).
Short when price breaks below daily L4 with volume > 2x 20-day avg AND weekly CCI < 100 (downtrend).
Avoids counter-trend trades in strong weekly trends, reducing false breakouts.
Designed for 1d timeframe to target 15-30 trades/year with high win rate in both bull/bear markets.
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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H4 and L4)
    camarilla_h4_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align daily levels to 1d timeframe (wait for daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Get weekly data for CCI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly CCI (20-period)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    tp_ma = typical_price.rolling(window=20, min_periods=20).mean()
    tp_md = typical_price.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    cci = (typical_price - tp_ma) / (0.015 * tp_md)
    cci_values = cci.values
    
    # Align weekly CCI to 1d timeframe
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
        
        # Long signal: break above daily H4 with volume filter AND weekly CCI > -100 (not strong downtrend)
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_filter[i] and 
                      cci_aligned[i] > -100)
        
        # Short signal: break below daily L4 with volume filter AND weekly CCI < 100 (not strong uptrend)
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

name = "1d_1W_Camarilla_Breakout_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0