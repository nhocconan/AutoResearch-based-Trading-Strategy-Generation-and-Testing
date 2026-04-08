#!/usr/bin/env python3
# 4h_1d_keltner_breakout_volume_v1
# Hypothesis: Trade Keltner channel breakouts on 4h with 1d trend filter and volume confirmation.
# In bull markets, buy breakouts above upper Keltner band with 1d uptrend; in bear markets, sell breakdowns below lower band with 1d downtrend.
# Volume surge confirms breakout strength. Uses ATR-based stops to manage risk.
# Target: 20-50 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA25/50 crossover
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h Keltner channels (20-period EMA + 2.0*ATR)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    upper_keltner = ema20 + 2.0 * atr
    lower_keltner = ema20 - 2.0 * atr
    
    # Volume confirmation: 4h volume > 2.5x 20-period average (stricter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema25_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below lower Keltner band OR stoploss hit
            if close[i] < lower_keltner[i] or close[i] < upper_keltner[i] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above upper Keltner band OR stoploss hit
            if close[i] > upper_keltner[i] or close[i] > lower_keltner[i] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above upper Keltner band with 1d uptrend and volume surge
            if (high[i] > upper_keltner[i-1] and  # New high breakout above previous upper band
                ema25_1d_aligned[i] > ema50_1d_aligned[i] and  # 1d uptrend
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Break below lower Keltner band with 1d downtrend and volume surge
            elif (low[i] < lower_keltner[i-1] and  # New low breakdown below previous lower band
                  ema25_1d_aligned[i] < ema50_1d_aligned[i] and  # 1d downtrend
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals