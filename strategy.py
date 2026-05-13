#!/usr/bin/env python3
name = "6h_KeltnerChannel_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1D data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1D for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D EMA to 6H timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(20) on 6H for Keltner Channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 20:
        atr[19] = np.nanmean(tr[1:20])
        for i in range(20, len(tr)):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Calculate EMA(20) on 6H for Keltner center line
    close_series = pd.Series(close)
    ema20_6h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel parameters
    kc_mult = 2.0
    upper_keltner = ema20_6h + kc_mult * atr
    lower_keltner = ema20_6h - kc_mult * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(ema20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: Price above/below 1D EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # LONG: Uptrend + break above upper Keltner channel
            if uptrend and close[i] > upper_keltner[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + break below lower Keltner channel
            elif downtrend and close[i] < lower_keltner[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below center line or trend reverses
            if close[i] < ema20_6h[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above center line or trend reverses
            if close[i] > ema20_6h[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals