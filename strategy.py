#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1S1_Breakout_Trend_Filter
Hypothesis: For 12h timeframe, use daily Camarilla pivot levels (R1, S1) as key support/resistance. 
Enter long when price breaks above R1 with volume confirmation and 1d trend filter (close > 1d EMA50). 
Enter short when price breaks below S1 with volume confirmation and 1d trend filter (close < 1d EMA50).
Exit when price returns to the daily pivot point (mean reversion) or on opposite breakout.
Uses volume surge (>1.5x 20-period average) to filter false breakouts. Designed for low-frequency, high-conviction trades.
"""

name = "12h_1D_Camarilla_R1S1_Breakout_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle first value
    prev_high[0] = df_1d['high'].iloc[0]
    prev_low[0] = df_1d['low'].iloc[0]
    prev_close[0] = df_1d['close'].iloc[0]
    
    # Camarilla equations
    range_ = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r2 = pivot + (range_ * 1.1 / 6)
    s2 = pivot - (range_ * 1.1 / 6)
    
    # Align daily levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1d trend filter: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(prices['volume']).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > (1.5 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(pivot_12h[i]) or np.isnan(ema50_12h[i]) or 
            np.isnan(vol_ma_12h[i])):
            if position != 0:
                # Maintain position until exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        if position == 0:
            # Look for breakout entries with trend and volume confirmation
            # Long: price breaks above R1, above 1d EMA50, with volume surge
            if (prices['close'].iloc[i] > r1_12h[i] and 
                prices['close'].iloc[i] > ema50_12h[i] and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 1d EMA50, with volume surge
            elif (prices['close'].iloc[i] < s1_12h[i] and 
                  prices['close'].iloc[i] < ema50_12h[i] and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Manage existing position
            if position == 1:
                # Long: exit when price returns to pivot (mean reversion) or breaks below S1 (stop)
                if prices['close'].iloc[i] <= pivot_12h[i]:
                    signals[i] = 0.0
                    position = 0
                elif prices['close'].iloc[i] < s1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit when price returns to pivot (mean reversion) or breaks above R1 (stop)
                if prices['close'].iloc[i] >= pivot_12h[i]:
                    signals[i] = 0.0
                    position = 0
                elif prices['close'].iloc[i] > r1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals