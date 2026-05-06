#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss
# Long when price breaks above 20-period Donchian high AND 1d close > 1d EMA50
# Short when price breaks below 20-period Donchian low AND 1d close < 1d EMA50
# Exit with ATR(14) trailing stop: signal→0 when price < highest_high_since_entry - 2.5*ATR (long) or
# price > lowest_low_since_entry + 2.5*ATR (short)
# Uses discrete sizing 0.30 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels provide robust structure in both trending and ranging markets
# 1d EMA50 filters for higher timeframe trend alignment with sufficient lag
# ATR stoploss manages risk during volatile periods like 2022 crash
# Works in both bull and bear markets by following the 1d trend

name = "4h_Donchian20_1dEMA50_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1d Donchian channels and EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donchian_high = high_1d_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) for 4h timeframe for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0
    lowest_since_entry = 0
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND uptrend (1d close > EMA50)
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short breakdown: price < Donchian low AND downtrend (1d close < EMA50)
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit if price drops below highest - 2.5*ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit if price rises above lowest + 2.5*ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals