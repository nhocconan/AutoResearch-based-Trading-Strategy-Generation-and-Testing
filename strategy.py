#!/usr/bin/env python3
"""
#100911 - 6h_Keltner_Channel_Breakout_1dTrend_Volume
Hypothesis: Breakouts from Keltner Channel (ATR-based) with daily trend filter and volume confirmation.
Keltner adapts to volatility, reducing false breakouts in ranging markets. Works in bull (breakouts with trend) 
and bear (mean reversion to middle band when trend weak). Targets 50-150 total trades over 4 years.
"""

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
    
    # Get 1d data for trend filter and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(10) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Keltner Channel: EMA(20) ± ATR(10)*2
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(ema20[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Keltner, above 1d EMA34, volume spike
        if (close[i] > upper_keltner[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Keltner, below 1d EMA34, volume spike
        elif (close[i] < lower_keltner[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle EMA (mean reversion when trend weak)
        elif position == 1 and close[i] < ema20[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema20[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Keltner_Channel_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0