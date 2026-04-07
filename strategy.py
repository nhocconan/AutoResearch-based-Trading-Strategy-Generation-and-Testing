#!/usr/bin/env python3
"""
4h_atr_breakout_12h_trend_volume_v2
Hypothesis: On 4h timeframe, enter long when price breaks above ATR-based upper band with above-average volume and 12h EMA21 uptrend, enter short when price breaks below ATR-based lower band with above-average volume and 12h EMA21 downtrend. Exit when price crosses opposite ATR band. Uses 12h EMA trend filter to avoid counter-trend trades. Designed for 20-50 trades/year to minimize fee drag while capturing breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_12h_trend_volume_v2"
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
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR bands (multiplier = 2.0)
    upper_band = close + 2.0 * atr
    lower_band = close - 2.0 * atr
    
    # Calculate 12h EMA21 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if data not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below lower band
            if close[i] < lower_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper band
            if close[i] > upper_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above upper band with 12h EMA21 uptrend
                if close[i] > upper_band[i] and ema_21_12h_aligned[i] > ema_21_12h_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower band with 12h EMA21 downtrend
                elif close[i] < lower_band[i] and ema_21_12h_aligned[i] < ema_21_12h_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals