#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based breakout with 1d trend filter and volume confirmation
# Uses ATR(14) to measure volatility and detect breakouts from Narrow Range 7 (NR7) patterns
# Breakouts occur when price moves beyond NR7 high/low + 0.5*ATR
# 1d EMA50 ensures we trade with higher timeframe trend to avoid whipsaws in choppy markets
# Volume confirmation (>1.5x 20-period EMA) validates breakout strength
# NR7 identifies low volatility contractions that often precede explosive moves
# Works in bull/bear markets: trend filter prevents counter-trend trades, breakouts capture momentum
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

name = "6h_ATR_NR7_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Narrow Range 7 (NR7) - lowest range of last 7 periods
    ranges = high - low
    rolling_min_range = pd.Series(ranges).rolling(window=7, min_periods=7).min().values
    is_nr7 = (ranges == rolling_min_range)
    
    # Calculate NR7 high and low breakout levels
    nr7_high = np.where(is_nr7, high, np.nan)
    nr7_low = np.where(is_nr7, low, np.nan)
    
    # Forward fill NR7 levels to use until next NR7
    nr7_high_series = pd.Series(nr7_high)
    nr7_high_ffill = nr7_high_series.ffill().values
    nr7_low_series = pd.Series(nr7_low)
    nr7_low_ffill = nr7_low_series.ffill().values
    
    # Breakout levels: NR7 high/low + 0.5*ATR
    breakout_up = nr7_high_ffill + 0.5 * atr
    breakout_down = nr7_low_ffill - 0.5 * atr
    
    # Volume confirmation: 20-period EMA on volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price breaks above NR7 high + 0.5*ATR + above 1d EMA50 + volume spike
            if (close[i] > breakout_up[i] and 
                close[i] > ema_50_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below NR7 low - 0.5*ATR + below 1d EMA50 + volume spike
            elif (close[i] < breakout_down[i] and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below NR7 low OR below 1d EMA50
            if close[i] < breakout_down[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above NR7 high OR above 1d EMA50
            if close[i] > breakout_up[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals