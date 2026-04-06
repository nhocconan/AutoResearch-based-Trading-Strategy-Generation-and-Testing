#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h trend filter.
# Long when price breaks above upper Donchian band with volume > 1.5x average and 12h EMA(20) rising.
# Short when price breaks below lower Donchian band with volume > 1.5x average and 12h EMA(20) falling.
# Uses ATR(14) for dynamic stoploss (2x ATR). Designed for fewer, high-quality trades in both bull and bear markets.

name = "4h_donchian20_vol_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below lower Donchian or 12h EMA turns down
            elif close[i] < lowest_low[i] or ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above upper Donchian or 12h EMA turns up
            elif close[i] > highest_high[i] or ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            if vol_strong[i]:
                # Long breakout: price breaks above upper Donchian with strong volume and rising 12h EMA
                if close[i] > highest_high[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below lower Donchian with strong volume and falling 12h EMA
                elif close[i] < lowest_low[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals