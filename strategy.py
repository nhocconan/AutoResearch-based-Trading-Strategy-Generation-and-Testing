#!/usr/bin/env python3
"""
4h_1d_Camarilla_Retest_Trend_Filter_v2
Hypothesis: Trade retests of daily Camarilla H3/L3 levels with volume confirmation and trend filter.
Long when price retests H3 as support in uptrend (close > EMA50), short when retests L3 as resistance in downtrend.
Exit on trend reversal or opposite level touch. Designed for low trade frequency (20-40/year) by requiring
retest confirmation and trend alignment, reducing false breakouts. Works in bull (continuation) and bear (mean reversion) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Retest_Trend_Filter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    range_1d = high_1d - low_1d
    
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # === TREND FILTER: 50 EMA ON 4H CHART ===
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    breakout_high = False  # Track if H3 was broken
    breakout_low = False   # Track if L3 was broken
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(ema50[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        # Volume strength (must be above average)
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Detect breakouts
        if not breakout_high and close[i] > h3_aligned[i]:
            breakout_high = True
        if not breakout_low and close[i] < l3_aligned[i]:
            breakout_low = True
            
        # Long: price retests H3 as support after breakout, with volume, in uptrend
        long_signal = (breakout_high and 
                      abs(close[i] - h3_aligned[i]) / h3_aligned[i] < 0.005 and  # Within 0.5% of H3
                      low[i] <= h3_aligned[i] and  # Price touched or went below H3 (retest)
                      close[i] > h3_aligned[i] and   # But closed back above H3
                      uptrend and 
                      strong_volume)
        
        # Short: price retests L3 as resistance after breakdown, with volume, in downtrend
        short_signal = (breakout_low and 
                       abs(close[i] - l3_aligned[i]) / l3_aligned[i] < 0.005 and  # Within 0.5% of L3
                       high[i] >= l3_aligned[i] and  # Price touched or went above L3 (retest)
                       close[i] < l3_aligned[i] and   # But closed back below L3
                       downtrend and 
                       strong_volume)
        
        # Exit: opposite H3/L3 level or trend reversal
        exit_long = (position == 1 and 
                    (close[i] < l3_aligned[i] or not uptrend))
        exit_short = (position == -1 and 
                     (close[i] > h3_aligned[i] or not downtrend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
            breakout_high = False  # Reset breakout flag after taking trade
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
            breakout_low = False   # Reset breakdown flag after taking trade
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
            breakout_high = False
            breakout_low = False
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
            breakout_high = False
            breakout_low = False
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals