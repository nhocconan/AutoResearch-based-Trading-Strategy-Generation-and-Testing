#!/usr/bin/env python3
# 12h_Keltner_Breakout_1wTrend_VolumeFilter
# Hypothesis: Price breaking above/below 2xATR Keltner Channels on 12h chart with 1-week EMA trend filter and volume confirmation.
# In bull markets, captures continuation of uptrends; in bear markets, captures breakdowns of downtrends.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaw.
# Volume confirmation filters out low-conviction breakouts. Designed for low trade frequency (~15-25/year) to minimize fee drag.

name = "12h_Keltner_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter (20-period)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate ATR for Keltner Channels (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Keltner Channels: 2 * ATR above/below 20-period EMA of close (on 12h)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_channel = ema_20 + 2 * atr
    lower_channel = ema_20 - 2 * atr
    
    # Volume confirmation (20-period MA on 12h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (20), ATR (14), EMA (20), volume MA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_20[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_long = close[i] > upper_channel[i]
        breakout_short = close[i] < lower_channel[i]
        
        if position == 0:
            # Long entry: price breaks above upper Keltner + weekly uptrend + volume spike
            if breakout_long and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Keltner + weekly downtrend + volume spike
            elif breakout_short and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA(20) or weekly trend turns down
            if close[i] < ema_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA(20) or weekly trend turns up
            if close[i] > ema_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals