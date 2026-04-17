#!/usr/bin/env python3
"""
1h_4h1d_Trend_Filter_MeanReversion
Strategy: Use 4h trend filter (price vs EMA34) and 1d mean reversion (distance from VWAP) for entries on 1h.
Long: 4h uptrend (price > EMA34) AND price < 1d VWAP - 1.5*ATR(20)
Short: 4h downtrend (price < EMA34) AND price > 1d VWAP + 1.5*ATR(20)
Exit: Price crosses 1d VWAP
Position size: 0.20
Session filter: 08-20 UTC only
Designed to capture mean reversion moves within the dominant 4h trend in both bull and bear markets.
Timeframe: 1h
"""

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
    volume = prices['volume'].values
    
    # Calculate 1h VWAP for exit
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate ATR(20) for entry threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate EMA34 on 4h close
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    # Calculate 1d VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num_1d = (typical_price_1d * volume_1d).cumsum()
    vwap_den_1d = volume_1d.cumsum()
    vwap_1d = np.divide(vwap_num_1d, vwap_den_1d, out=np.full_like(vwap_num_1d, np.nan), where=vwap_den_1d!=0)
    # Align 1d VWAP to 1h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # EMA34 and ATR20
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h
        uptrend_4h = close[i] > ema34_4h_aligned[i]
        downtrend_4h = close[i] < ema34_4h_aligned[i]
        
        # Mean reversion signals from 1d VWAP
        price_vwap_diff = close[i] - vwap_1d_aligned[i]
        oversold = price_vwap_diff < (-1.5 * atr[i])  # price significantly below VWAP
        overbought = price_vwap_diff > (1.5 * atr[i])  # price significantly above VWAP
        
        # Exit condition: price crosses 1d VWAP
        cross_vwap = (position == 1 and close[i] < vwap_1d_aligned[i]) or \
                     (position == -1 and close[i] > vwap_1d_aligned[i])
        
        if position == 0:
            # Long: 4h uptrend AND oversold relative to 1d VWAP
            if uptrend_4h and oversold:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend AND overbought relative to 1d VWAP
            elif downtrend_4h and overbought:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses VWAP or trend changes
            if cross_vwap or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses VWAP or trend changes
            if cross_vwap or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Trend_Filter_MeanReversion"
timeframe = "1h"
leverage = 1.0