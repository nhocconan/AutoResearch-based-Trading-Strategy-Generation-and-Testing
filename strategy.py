#!/usr/bin/env python3
"""
1h_Pullback_EMA_Trend
Hypothesis: In 1h timeframe, price pulls back to EMA20 during strong trends (4h/1d aligned). 
Go long when price touches EMA20 from below in uptrend (4h close > EMA50, 1d close > EMA50) with volume confirmation.
Go short when price touches EMA20 from above in downtrend (4h close < EMA50, 1d close < EMA50) with volume confirmation.
Exit on opposite touch. Position size: ±0.20. Uses EMA20 for dynamic support/resistance.
Designed to work in bull (buy pullbacks) and bear (sell rallies) by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA20 for dynamic support/resistance
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    close_series_4h = pd.Series(close_4h)
    ema50_4h = close_series_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 50, 50, 20)  # EMA20, EMA50_4h, EMA50_1d, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Determine trend alignment: both 4h and 1d must agree
        uptrend = (close_4h[i // 16] > ema50_4h[i // 16]) and (close_1d[i // 384] > ema50_1d[i // 384])
        downtrend = (close_4h[i // 16] < ema50_4h[i // 16]) and (close_1d[i // 384] < ema50_1d[i // 384])
        
        # Price touching EMA20 conditions
        touch_from_below = low[i] <= ema20[i] and close[i] > ema20[i]
        touch_from_above = high[i] >= ema20[i] and close[i] < ema20[i]
        
        if position == 0:
            # Long: touch EMA20 from below + volume filter + uptrend on both 4h and 1d
            if touch_from_below and volume_filter and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: touch EMA20 from above + volume filter + downtrend on both 4h and 1d
            elif touch_from_above and volume_filter and downtrend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: touch EMA20 from above
            if touch_from_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: touch EMA20 from below
            if touch_from_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Pullback_EMA_Trend"
timeframe = "1h"
leverage = 1.0