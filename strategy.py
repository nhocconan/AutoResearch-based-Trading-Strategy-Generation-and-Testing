#!/usr/bin/env python3
"""
1d_keltner_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, enter long when price closes above upper Keltner channel (EMA20 + 2*ATR) with above-average volume and 1w EMA trend up, enter short when price closes below lower Keltner channel (EMA20 - 2*ATR) with above-average volume and 1w EMA trend down. Exit when price crosses EMA20 (mean reversion). Designed for 10-30 trades/year to minimize fee decay while capturing volatility expansion moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA20 (middle of Keltner)
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate ATR for Keltner width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Keltner channels
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 1w EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(close[i]) or np.isnan(ema_20[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20 (mean reversion)
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20 (mean reversion)
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price closes above upper Keltner with 1w EMA trending up
                if close[i] > upper_keltner[i] and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price closes below lower Keltner with 1w EMA trending down
                elif close[i] < lower_keltner[i] and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals