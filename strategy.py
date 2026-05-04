#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR(14) volatility filter
# Uses discrete position sizing (0.30) to minimize fee churn. Donchian channels provide robust
# price channels that work in both trending and ranging markets. 1d EMA34 ensures alignment with
# higher-timeframe trend, while ATR filter avoids low-volatility false breakouts. Target: 20-30 trades/year per symbol.

name = "4h_Donchian20_1dEMA34_ATRFilter_Trend"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for ATR(14) volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # first bar: no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR filter: avoid low volatility conditions (ATR < 0.5 * 20-period average)
        atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().iloc[i] if i >= 20 else atr_14_1d_aligned[i]
        low_volatility = atr_14_1d_aligned[i] < (0.5 * atr_ma)
        
        if position == 0:
            # Long: price breaks above Donchian upper + 1d EMA34 bullish + not low volatility
            if (close[i] > highest_20[i] and close[i] > ema_34_1d_aligned[i] and not low_volatility):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower + 1d EMA34 bearish + not low volatility
            elif (close[i] < lowest_20[i] and close[i] < ema_34_1d_aligned[i] and not low_volatility):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian lower OR 1d EMA34 turns bearish
            if close[i] < lowest_20[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above Donchian upper OR 1d EMA34 turns bullish
            if close[i] > highest_20[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals