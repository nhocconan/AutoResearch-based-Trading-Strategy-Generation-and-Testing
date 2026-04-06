#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with weekly EMA200 trend filter and volume confirmation
# Uses 12-period Donchian channels for breakout detection, 200-period EMA on weekly for trend filter
# Requires volume > 1.5x 12-period average for confirmation
# Implements 2x ATR(14) stoploss to manage risk
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drift
# Works in both bull and bear markets by following the higher timeframe trend

name = "12h_donchian12_1w_ema200_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # EMA200 calculation on weekly
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align weekly EMA200 to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 12h Donchian channels (12-period)
    # Highest high of last 12 periods (excluding current)
    donchian_high = pd.Series(high).rolling(window=12, min_periods=12).max().shift(1).values
    # Lowest low of last 12 periods (excluding current)
    donchian_low = pd.Series(low).rolling(window=12, min_periods=12).min().shift(1).values
    
    # Volume average (12-period)
    volume_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or trend turns down
            elif close[i] < donchian_low[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or trend turns up
            elif close[i] > donchian_high[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above Donchian high, weekly uptrend, volume spike
            if (close[i] > donchian_high[i] and
                close[i] > ema200_1w_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, weekly downtrend, volume spike
            elif (close[i] < donchian_low[i] and
                  close[i] < ema200_1w_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals