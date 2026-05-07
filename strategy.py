#!/usr/bin/env python3
# 4h_ADX_Trend_Strength_Volume_Signal
# Hypothesis: Uses ADX(14) > 25 to identify strong trends, combined with volume confirmation (volume > 1.5x 20-period average) and price position relative to 50-period EMA. In strong trends, price tends to continue in the direction of the EMA. Works in both bull and bear markets by following the trend direction only when ADX confirms trend strength, avoiding whipsaws in ranging markets. Target: 20-30 trades/year per symbol to minimize fee drag.

timeframe = "4h"
name = "4h_ADX_Trend_Strength_Volume_Signal"
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
    
    # Calculate ADX components
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) 
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    plus_dm_period = wilders_smoothing(plus_dm, period)
    minus_dm_period = wilders_smoothing(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # EMA50 for trend direction
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 2*period)  # Ensure we have EMA50, volume MA, and ADX data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(adx[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend), volume > 1.5x average, price above EMA50
            if (adx[i] > 25 and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend), volume > 1.5x average, price below EMA50
            elif (adx[i] > 25 and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend weakening (ADX < 20) or price crosses below EMA50
            if adx[i] < 20 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakening (ADX < 20) or price crosses above EMA50
            if adx[i] < 20 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals