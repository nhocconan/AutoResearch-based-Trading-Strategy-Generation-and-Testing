#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_With_Flow
Hypothesis: Breakout above H3 or below L3 of daily Camarilla levels with 12-hour trend alignment and volume confirmation. 
Long when price breaks above H3 with 12h uptrend and volume surge; short when breaks below L3 with 12h downtrend and volume surge.
Uses 12-hour timeframe for signal generation with daily Camarilla levels for institutional support/resistance.
Designed to capture momentum moves in both bull and bear markets with controlled trade frequency.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_With_Flow"
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
    
    # === 1-DAY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            camarilla_h3[i] = camarilla_l3[i] = camarilla_h4[i] = camarilla_l4[i] = np.nan
            continue
            
        range_val = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + range_val * 1.1 / 6
        camarilla_l3[i] = close_1d[i] - range_val * 1.1 / 6
        camarilla_h4[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 12-HOUR TREND FILTER ===
    # Calculate EMA25 of 12-hour close for trend
    ema25_12h = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):
        # Skip if not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(ema25_12h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h trend: up if close > EMA25, down if close < EMA25
        trend_up = close[i] > ema25_12h[i]
        trend_down = close[i] < ema25_12h[i]
        
        # Long: breakout above H3 in uptrend with volume surge
        long_signal = (trend_up and 
                      close[i] > h3_12h[i] and 
                      vol_ratio[i] > 1.8)
        
        # Short: breakout below L3 in downtrend with volume surge
        short_signal = (trend_down and 
                       close[i] < l3_12h[i] and 
                       vol_ratio[i] > 1.8)
        
        # Exit conditions: reversal of trend or price moves back to opposite H3/L3 level
        exit_long = (position == 1 and 
                    (not trend_up or close[i] < l3_12h[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] > h3_12h[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals