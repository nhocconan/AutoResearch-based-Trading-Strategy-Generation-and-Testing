#!/usr/bin/env python3
"""
1h_atr_breakout_4h1d_trend_volume_v1
Hypothesis: ATR-based breakout strategy with 4h/1d trend filter and volume confirmation.
ATR(14) breakout captures momentum in both bull and bear markets. 4h EMA50 and 1d EMA100
provide multi-timeframe trend alignment. Volume > 20-period average confirms breakout strength.
Entry only during 08-20 UTC to avoid low-volume overnight hours. Target 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_atr_breakout_4h1d_trend_volume_v1"
timeframe = "1h"
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
    
    # 1h ATR for breakout levels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA100 for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 20-period volume average for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available or outside session
        if (np.isnan(atr[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema100_1d_aligned[i]) or np.isnan(vol_sma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        # Calculate dynamic breakout levels
        upper_break = high[i-1] + 0.5 * atr[i-1]
        lower_break = low[i-1] - 0.5 * atr[i-1]
        
        if position == 1:  # Long position
            # Exit: close below 1h EMA20 or ATR trailing stop
            ema20 = pd.Series(close[i-19:i+1]).mean() if i >= 19 else ema50_4h_aligned[i]
            if close[i] < ema20 or close[i] < (high[i-1] - 1.5 * atr[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: close above 1h EMA20 or ATR trailing stop
            ema20 = pd.Series(close[i-19:i+1]).mean() if i >= 19 else ema50_4h_aligned[i]
            if close[i] > ema20 or close[i] > (low[i-1] + 1.5 * atr[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long breakout in uptrend (price > 4h EMA50 and 1d EMA100)
            if (close[i] > upper_break and 
                vol_confirm and 
                close[i] > ema50_4h_aligned[i] and 
                close[i] > ema100_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short breakout in downtrend (price < 4h EMA50 and 1d EMA100)
            elif (close[i] < lower_break and 
                  vol_confirm and 
                  close[i] < ema50_4h_aligned[i] and 
                  close[i] < ema100_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals