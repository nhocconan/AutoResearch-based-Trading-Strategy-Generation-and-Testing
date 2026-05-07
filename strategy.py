#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below Keltner Channel (20, 1.5) on 4h with 1d EMA34 trend filter and volume spike (1.5x average) captures institutional breakouts. Designed for 4h to balance trade frequency and performance, avoiding excessive trades that cause fee drag while maintaining edge in both bull and bear markets by following higher timeframe trend.
"""
name = "4h_Keltner_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 and ATR10 for Keltner Channel (20-period EMA, 10-period ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(high - low).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channels
    upper_keltner = ema_20 + (1.5 * atr_10)
    lower_keltner = ema_20 - (1.5 * atr_10)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 24-period average (balanced)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20[i]) or np.isnan(atr_10[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner + 1d uptrend + volume spike
            if close[i] > upper_keltner[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + 1d downtrend + volume spike
            elif close[i] < lower_keltner[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to middle Keltner line (EMA20)
            if position == 1:
                if close[i] <= ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals