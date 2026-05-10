#!/usr/bin/env python3
# 12h_WR_Trend_With_1wTrend
# Hypothesis: Williams %R on 12h timeframe combined with 1-week EMA trend filter provides robust entries in both bull and bear markets. 
# Williams %R identifies overbought/oversold conditions for mean reversion entries, while the weekly EMA ensures we trade with the higher timeframe trend. 
# Volume confirmation filters out low-conviction signals. This combination should work in ranging markets (via mean reversion) and trending markets (via trend alignment).

name = "12h_WR_Trend_With_1wTrend"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Williams %R on 12h chart (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-period MA on 12h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Williams %R (14), weekly EMA (50), volume MA (20)
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(wr[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Williams %R conditions
        wr_oversold = wr[i] < -80  # Oversold condition for long
        wr_overbought = wr[i] > -20  # Overbought condition for short
        
        if position == 0:
            # Long entry: uptrend + Williams %R oversold + volume
            if uptrend and wr_oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + Williams %R overbought + volume
            elif downtrend and wr_overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or Williams %R returns from oversold
            if not uptrend or wr[i] > -50:  # Exit when WR returns above -50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or Williams %R returns from overbought
            if not downtrend or wr[i] < -50:  # Exit when WR returns below -50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals