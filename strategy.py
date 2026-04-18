#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_Volume_Trend_v1
Hypothesis: Use 1d EMA50 for trend direction and 4h Keltner Channel breakout with volume confirmation. 
Go long when price breaks above Keltner upper band AND price > 1d EMA50, short when price breaks below Keltner lower band AND price < 1d EMA50. 
Requires volume > 1.3x 20-period average for confirmation. Uses ATR-based stoploss via signal reversal. 
Designed for 4-6 trades/month (~50-80/year) to avoid fee drag. Works in bull via trend-following longs and in bear via trend-following shorts.
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
    
    # Get 4h data for Keltner Channel (ATR-based)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h ATR(10) for Keltner Channel
    atr_period = 10
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])  # align length
    atr_4h = np.full_like(close_4h, np.nan)
    
    if len(tr_4h) >= atr_period:
        for i in range(atr_period, len(tr_4h)):
            atr_4h[i] = np.nanmean(tr_4h[i-atr_period+1:i+1])
    
    # 4h EMA20 for Keltner Channel middle
    ema_period = 20
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_4h[ema_period-1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] - ema_4h[i-1]) * multiplier + ema_4h[i-1]
    
    # Keltner Channel: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_mult = 2.0
    upper_4h = ema_4h + keltner_mult * atr_4h
    lower_4h = ema_4h - keltner_mult * atr_4h
    
    # Align Keltner Channels to 4h timeframe (same timeframe, no alignment needed)
    upper_4h_aligned = upper_4h
    lower_4h_aligned = lower_4h
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50
    ema50_period = 50
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema50_period:
        multiplier = 2 / (ema50_period + 1)
        ema50_1d[ema50_period-1] = np.mean(close_1d[:ema50_period])
        for i in range(ema50_period, len(close_1d)):
            ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * multiplier + ema50_1d[i-1]
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, ema_period, vol_period, ema50_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner upper AND price > 1d EMA50 + volume
            if close[i] > upper_4h_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > 1.3 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner lower AND price < 1d EMA50 + volume
            elif close[i] < lower_4h_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > 1.3 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Keltner lower (reverse to short)
            if close[i] < lower_4h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Keltner upper (reverse to long)
            if close[i] > upper_4h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Channel_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0