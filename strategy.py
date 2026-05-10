#!/usr/bin/env python3
# 4h_12h_PriceChannel_Breakout
# Hypothesis: Combines 12h price channel breakout with 4h trend filter and volume confirmation
# Works in both bull and bear markets by capturing breakouts aligned with higher timeframe trend
# Price channel acts as support/resistance structure, reducing whipsaw
# Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_12h_PriceChannel_Breakout"
timeframe = "4h"
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
    
    # 12h price channel (20-period high/low)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period high/low channels
    high_ch = np.full(len(high_12h), np.nan)
    low_ch = np.full(len(low_12h), np.nan)
    
    for i in range(20, len(high_12h)):
        high_ch[i] = np.max(high_12h[i-20:i])
        low_ch[i] = np.min(low_12h[i-20:i])
    
    # Align 12h channels to 4h timeframe
    high_ch_aligned = align_htf_to_ltf(prices, df_12h, high_ch)
    low_ch_aligned = align_htf_to_ltf(prices, df_12h, low_ch)
    
    # 4h trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_4h_up = close_4h > ema34_4h
    trend_4h_down = close_4h < ema34_4h
    
    # Align 4h trend to 4h (no shift needed as we're on same timeframe)
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_ch_aligned[i]) or np.isnan(low_ch_aligned[i]) or
            np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h high channel with volume confirmation and 4h uptrend
            if (high[i] > high_ch_aligned[i] and
                trend_4h_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h low channel with volume confirmation and 4h downtrend
            elif (low[i] < low_ch_aligned[i] and
                  trend_4h_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 12h low channel or 4h trend turns down
            if (low[i] < low_ch_aligned[i] or
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above 12h high channel or 4h trend turns up
            if (high[i] > high_ch_aligned[i] or
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals