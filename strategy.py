#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(34) for trend
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for ATR and volatility (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate true range and ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d - low_1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-period ATR multiplier for breakout threshold
    atr_mult = 0.5
    
    # Calculate 12-period high/low for breakout levels
    high_roll = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_roll = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need EMA(34) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above 12-period high with volume and trend filter
            if close[i] > high_roll[i] and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 12-period low with volume and trend filter
            elif close[i] < low_roll[i] and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below weekly EMA34 OR ATR-based trailing stop
            if close[i] < ema_34_1w_aligned[i] or close[i] < (high_roll[i] - atr_mult * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above weekly EMA34 OR ATR-based trailing stop
            if close[i] > ema_34_1w_aligned[i] or close[i] > (low_roll[i] + atr_mult * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA34_Breakout_ATR_VolumeFilter"
timeframe = "12h"
leverage = 1.0