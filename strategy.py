#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume spike
# Why it should work in both bull AND bear:
# - Donchian breakouts capture strong directional moves regardless of trend direction
# - EMA200 on 1d filters against counter-trend moves (bull: only long above EMA200, bear: only short below EMA200)
# - Volume spike confirms institutional participation (not retail noise)
# - Tight entry conditions (~100 train trades total) minimize fee drag
# - Position size 0.25 manages drawdown (77% BTC crash → only 19% equity loss)
# Target: 75-150 total trades over 4 years (19-37/year), hard max 200

name = "4h_donchian20_1d_ema200_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (CRITICAL: avoid file I/O in loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # 4h volume average (20-bar rolling mean)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_1d_200_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Stoploss check (apply to both long and short)
        if position == 1:
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Exit conditions for active positions
        if position == 1:
            if close[i] < ema_1d_200_aligned[i]:  # Price below 1d EMA200 → trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > ema_1d_200_aligned[i]:  # Price above 1d EMA200 → trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Entry conditions: tight filter for quality signals
            # Long: price breaks above 4h Donchian upper, price > 1d EMA200 (uptrend), volume spike
            if (close[i] > donchian_upper[i-1] and
                close[i] > ema_1d_200_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below 4h Donchian lower, price < 1d EMA200 (downtrend), volume spike
            elif (close[i] < donchian_lower[i-1] and
                  close[i] < ema_1d_200_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals