#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian high(20), weekly close > weekly pivot, volume > 1.5x average
# Short when price breaks below Donchian low(20), weekly close < weekly pivot, volume > 1.5x average
# Exit when price re-enters Donchian channel or weekly trend changes
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses weekly pivot for trend filter and Donchian for breakout signals
# Target: 50-150 total trades over 4 years (12-38/year)

name = "6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly data for pivot and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly pivot point: (H + L + C) / 3
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    
    # Weekly trend: close > pivot = uptrend, close < pivot = downtrend
    weekly_trend = (close_weekly > pivot_weekly).astype(int)  # 1 for uptrend, 0 for downtrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend)
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pivot_weekly_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel or weekly trend turns down
            elif close[i] <= donch_high[i] or weekly_trend_aligned[i] == 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel or weekly trend turns up
            elif close[i] >= donch_low[i] or weekly_trend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation and weekly trend alignment
            bullish_breakout = close[i] > donch_high[i-1]  # break above previous high
            bearish_breakout = close[i] < donch_low[i-1]   # break below previous low
            
            # Long: bullish breakout, weekly uptrend, volume spike
            if (bullish_breakout and
                weekly_trend_aligned[i] == 1 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, weekly downtrend, volume spike
            elif (bearish_breakout and
                  weekly_trend_aligned[i] == 0 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals