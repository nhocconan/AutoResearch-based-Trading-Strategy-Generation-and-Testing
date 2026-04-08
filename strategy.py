#!/usr/bin/env python3
"""
1d Camarilla pivot with 1w trend filter and volume confirmation
Hypothesis: Price rejecting Camarilla pivot levels (H3/L3) in direction of 1w EMA(50) trend with volume confirmation captures reversals in ranging markets and continuations in trending markets. 1w trend filter ensures alignment with higher timeframe momentum. Target: 10-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Previous day's OHLC for Camarilla calculation (using daily data)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels: H3/L3 = C +- (H-L)*1.1/2
    rang = prev_high - prev_low
    h3 = prev_close + rang * 1.1 / 2
    l3 = prev_close - rang * 1.1 / 2
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(h3[i]) or 
            np.isnan(l3[i]) or 
            np.isnan(vol_surge[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR price breaks below L3
            if (close[i] <= ema_50_1w_aligned[i] or 
                close[i] < l3[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price breaks above H3
            if (close[i] >= ema_50_1w_aligned[i] or 
                close[i] > h3[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price rebounds from L3 + volume surge + uptrend
            if (close[i] > l3[i] and
                close[i] < h3[i] and  # within H3/L3 range
                close[i] > ema_50_1w_aligned[i] and
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short: price rejects from H3 + volume surge + downtrend
            elif (close[i] < h3[i] and
                  close[i] > l3[i] and  # within H3/L3 range
                  close[i] < ema_50_1w_aligned[i] and
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals