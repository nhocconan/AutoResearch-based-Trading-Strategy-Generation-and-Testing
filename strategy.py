#!/usr/bin/env python3
# 1h_4h_1d_VWAP_Breakout_Trend
# Hypothesis: 1-hour price breaking above/below the volume-weighted average price (VWAP) of the prior 4-hour candle, 
# confirmed by 1-day trend (price above/below EMA200) and volume surge, captures intraday momentum with institutional validation.
# VWAP acts as dynamic support/resistance; breakouts with volume indicate strong directional moves.
# Works in bull markets by buying VWAP breaks in uptrends; in bear markets by selling VWAP breaks in downtrends.
# Volume filter reduces false signals; trend filter ensures alignment with higher timeframe bias.
# Target: 20-40 trades/year to stay within fee limits.

name = "1h_4h_1d_VWAP_Breakout_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for VWAP calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for each 4h bar: typical price * volume / cumulative volume
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    vwap_4h = (typical_price_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h = vwap_4h.values
    
    # Align 4h VWAP to 1h timeframe (wait for 4h bar to close)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA200 on 1d close
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 1h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: volume > 1.5x 20-period EMA on 1h
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure EMA200 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above 4h VWAP + uptrend (price > EMA200) + volume surge
            if close[i] > vwap_4h_aligned[i] and close[i] > ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price crosses below 4h VWAP + downtrend (price < EMA200) + volume surge
            elif close[i] < vwap_4h_aligned[i] and close[i] < ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses back below VWAP or trend changes
            if close[i] < vwap_4h_aligned[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price crosses back above VWAP or trend changes
            if close[i] > vwap_4h_aligned[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals