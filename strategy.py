#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) stoploss.
# Long when price breaks above Donchian upper (20-period high) in 1d uptrend (price > EMA50).
# Short when price breaks below Donchian lower (20-period low) in 1d downtrend (price < EMA50).
# Uses ATR-based trailing stop: exit long if price drops 2.0*ATR from highest high since entry.
# Exit short if price rises 2.0*ATR from lowest low since entry.
# Volume confirmation: current volume > 1.5x 20-period MA to avoid false breakouts.
# Discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
# Donchian channels provide objective structure; 1d EMA50 ensures higher timeframe alignment.
# ATR stoploss manages risk without look-ahead. Works in both bull and bear markets by
# only trading with the 1d trend, reducing whipsaw during ranging periods.

name = "4h_Donchian20_1dEMA50_ATR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Donchian upper AND 1d uptrend AND volume spike
            if close_val > highest_high[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            # Short: price breaks below Donchian lower AND 1d downtrend AND volume spike
            elif close_val < lowest_low[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Update highest high since entry
            if close_val > highest_since_entry:
                highest_since_entry = close_val
            # Long exit: price drops 2.0*ATR from highest high since entry
            if close_val < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            if close_val < lowest_since_entry:
                lowest_since_entry = close_val
            # Short exit: price rises 2.0*ATR from lowest low since entry
            if close_val > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals