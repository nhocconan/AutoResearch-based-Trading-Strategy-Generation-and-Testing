#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR breakout with volume spike and 1d trend filter.
# Uses 1d EMA for trend direction and ATR-based breakout levels.
# Designed to work in both bull and bear markets by following 1d trend.
# Target: 12-37 trades/year per symbol to avoid fee drag.
name = "6h_ATR_Breakout_Volume_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average for spike detection
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # ATR (14) on 6h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Breakout levels: ATR multiplier
    atr_mult = 1.5
    upper_breakout = np.roll(close, 1) + atr_mult * atr
    lower_breakout = np.roll(close, 1) - atr_mult * atr
    
    # Volume spike: current volume > 1.5x 34-period EMA
    vol_ema_6h = pd.Series(volume).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_breakout[i]) or 
            np.isnan(lower_breakout[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper breakout with volume spike in uptrend
            long_condition = (close[i] > upper_breakout[i]) and vol_spike[i] and uptrend
            # Short breakdown: price < lower breakout with volume spike in downtrend
            short_condition = (close[i] < lower_breakout[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below upper breakout or trend turns down
            if (close[i] < upper_breakout[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above lower breakout or trend turns up
            if (close[i] > lower_breakout[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals