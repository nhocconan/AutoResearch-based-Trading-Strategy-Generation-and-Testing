#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and ATR(14) volatility filter.
# Long when price breaks above Donchian upper band, close > 1d EMA34, and ATR ratio > 0.8.
# Short when price breaks below Donchian lower band, close < 1d EMA34, and ATR ratio > 0.8.
# ATR ratio = current ATR(14) / 50-period ATR average to filter low-volatility breakouts.
# Works in bull markets via trend-following breakouts and in bear markets via volatility expansion signals.
# Discrete sizing 0.25 targets 50-150 total trades over 4 years on 6h timeframe.

name = "6h_Donchian20_1dEMA34_Trend_ATRFilter"
timeframe = "6h"
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
    
    # Calculate Donchian channels (20-period)
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility filter
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 50-period average ATR for volatility regime filter
    avg_atr = pd.Series(atr).rolling(window=50, min_periods=50).mean().shift(1).values
    atr_ratio = atr / np.where(avg_atr > 0, avg_atr, np.nan)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback_dc, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 1d EMA34, sufficient volatility
            if (high[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                atr_ratio[i] > 0.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 1d EMA34, sufficient volatility
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  atr_ratio[i] > 0.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower OR volatility drops
            if (low[i] < lowest_low[i] or 
                atr_ratio[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper OR volatility drops
            if (high[i] > highest_high[i] or 
                atr_ratio[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals