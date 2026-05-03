#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
# Long when price breaks above 4h Donchian upper channel in 1d uptrend (price > EMA50).
# Short when price breaks below 4h Donchian lower channel in 1d downtrend (price < EMA50).
# ATR stoploss: exit long if price drops 2.5*ATR below highest high since entry.
# Exit short if price rises 2.5*ATR above lowest low since entry.
# Uses discrete sizing 0.30 to balance return and drawdown. Target: 75-200 total trades over 4 years.
# Donchian channels provide clear structure, 1d EMA50 ensures higher timeframe alignment,
# ATR stoploss manages risk without look-ahead. Works in both bull and bear markets by
# only trading with the 1d trend, avoiding counter-trend whipsaws.

name = "4h_Donchian20_1dEMA50_ATRStop_Volume"
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
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above Donchian high AND 1d uptrend
            if close_val > donchian_high[i] and trend_up:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                highest_high = close_val
            # Short: price breaks below Donchian low AND 1d downtrend
            elif close_val < donchian_low[i] and trend_down:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                lowest_low = close_val
        elif position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, close_val)
            # ATR trailing stop: exit if price drops 2.5*ATR below highest high
            if close_val < highest_high - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, close_val)
            # ATR trailing stop: exit if price rises 2.5*ATR above lowest low
            if close_val > lowest_low + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals