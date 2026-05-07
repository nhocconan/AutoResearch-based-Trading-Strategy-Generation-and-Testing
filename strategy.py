#!/usr/bin/env python3
"""
6h_Keltner_Breakout_WTrend_Volume
Hypothesis: Keltner Channel breakouts with weekly trend filter and volume confirmation capture momentum with reduced false signals. 
In bull markets: price breaks above upper Keltner band with weekly uptrend = long. 
In bear markets: price breaks below lower Keltner band with weekly downtrend = short. 
The Keltner Channel (ATR-based) adapts to volatility, providing dynamic support/resistance. 
Weekly trend filter ensures alignment with higher timeframe momentum. 
Volume confirmation adds conviction. 
Low frequency via 6h timeframe and strict entry criteria (Keltner breakout + weekly trend + volume).
Target: 50-150 total trades over 4 years.
"""
name = "6h_Keltner_Breakout_WTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0)
    # Middle line: 20-period EMA of close
    close_s = pd.Series(close)
    middle = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Average True Range (ATR) for band width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower bands
    upper = middle + (2.0 * atr)
    lower = middle - (2.0 * atr)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(middle[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band + weekly uptrend + volume
            if close[i] > upper[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + weekly downtrend + volume
            elif close[i] < lower[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to middle line (mean reversion to average)
            if position == 1:
                if close[i] <= middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals