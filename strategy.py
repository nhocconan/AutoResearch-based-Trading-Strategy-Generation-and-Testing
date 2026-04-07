#!/usr/bin/env python3
"""
4h_atr_breakout_1d_trend_volume_v1
Hypothesis: ATR-based breakout with daily EMA trend filter and volume confirmation.
Buy when price breaks above ATR-based upper band in uptrend (price > EMA200),
sell when price breaks below ATR-based lower band in downtrend (price < EMA200).
Volume confirmation reduces false breakouts. Works in both bull and bear markets
by adapting to daily trend via EMA200 filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # ATR calculation (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR-based bands (2.5 * ATR)
    atr_mult = 2.5
    upper_band = close + atr_mult * atr
    lower_band = close - atr_mult * atr
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema200_4h[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        vol_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower band OR EMA trend turns down
            if close[i] < lower_band[i] or close[i] < ema200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper band OR EMA trend turns up
            if close[i] > upper_band[i] or close[i] > ema200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout in uptrend (price > EMA200)
            if (close[i] > upper_band[i] and 
                vol_confirm and 
                close[i] > ema200_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown in downtrend (price < EMA200)
            elif (close[i] < lower_band[i] and 
                  vol_confirm and 
                  close[i] < ema200_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals