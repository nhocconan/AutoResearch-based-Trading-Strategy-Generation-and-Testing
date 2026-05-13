#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) volatility filter.
# Long when price breaks above upper Donchian channel, close > 1d EMA50, and ATR(14) > 0.5 * ATR(50) (ensuring sufficient volatility).
# Short when price breaks below lower Donchian channel, close < 1d EMA50, and ATR(14) > 0.5 * ATR(50).
# Uses ATR-based stoploss: exit long if price drops below highest high since entry - 2.0 * ATR(14).
# Exit short if price rises above lowest low since entry + 2.0 * ATR(14).
# Designed for low trade frequency (<100 total 4h trades) to minimize fee drag while capturing strong momentum moves.

name = "4h_Donchian20_1dEMA50_ATRVol_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    lookback_dc = 20
    upper_dc = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    lower_dc = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    lookback_atr1 = 14
    lookback_atr2 = 50
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    atr14 = pd.Series(tr1).rolling(window=lookback_atr1, min_periods=lookback_atr1).mean().values
    tr2 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr2[0] = high[0] - low[0]
    atr50 = pd.Series(tr2).rolling(window=lookback_atr2, min_periods=lookback_atr2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(max(lookback_dc, lookback_atr2, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(atr50[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) > 0.5 * ATR(50)
        vol_filter = atr14[i] > 0.5 * atr50[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian, close > 1d EMA50, sufficient volatility
            if (high[i] > upper_dc[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
            # SHORT: Price breaks below lower Donchian, close < 1d EMA50, sufficient volatility
            elif (low[i] < lower_dc[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: Price drops below highest high since entry - 2.0 * ATR(14)
            if low[i] < highest_since_entry[i] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: Price rises above lowest low since entry + 2.0 * ATR(14)
            if high[i] > lowest_since_entry[i] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals