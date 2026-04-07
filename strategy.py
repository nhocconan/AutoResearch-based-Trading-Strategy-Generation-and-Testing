#!/usr/bin/env python3
"""
4h_atr_breakout_1d_trend_volume_v3
Hypothesis: On 4h timeframe, enter long when price breaks above ATR-based upper channel with volume confirmation and 1d EMA trend filter, enter short when price breaks below ATR-based lower channel with volume confirmation and 1d EMA trend filter. Exit on opposite ATR band touch or trend reversal. Designed for 20-40 trades/year to minimize fee dust while capturing breakouts in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v3"
timeframe = "4h"
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
    
    # Calculate ATR (14-period)
    if len(close) < 14:
        return np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-based channels (multiplier = 2.0)
    atr_mult = 2.0
    upper_channel = close + atr_mult * atr
    lower_channel = close - atr_mult * atr
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower channel OR trend turns bearish
            if close[i] <= lower_channel[i] or ema_1d_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper channel OR trend turns bullish
            if close[i] >= upper_channel[i] or ema_1d_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above upper channel with bullish 1d trend
                if close[i] > upper_channel[i] and ema_1d_aligned[i] > close[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower channel with bearish 1d trend
                elif close[i] < lower_channel[i] and ema_1d_aligned[i] < close[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals