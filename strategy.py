#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA40 trend filter and volume confirmation
# Breakouts above 20-day high (long) or below 20-day low (short) are taken only when
# aligned with weekly trend (price > EMA40 for longs, < EMA40 for shorts).
# Volume > 1.5x 20-day average confirms breakout strength.
# Target: 30-100 total trades over 4 years (7-25/year) with strict entry conditions.

name = "1d_donchian20_weekly_ema40_vol_v1"
timeframe = "1d"
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
    
    # 1d data for Donchian channels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for additional filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # 1w data for EMA40 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40 = pd.Series(close_1w).ewm(span=40, adjust=False).mean().values
    ema_40_aligned = align_htf_to_ltf(prices, df_1w, ema_40)
    
    # 1d volume average for confirmation
    volume_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50[i]) or
            np.isnan(ema_40_aligned[i]) or np.isnan(volume_ma_20_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel
            elif close[i] < high_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel
            elif close[i] > low_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: breakout above 20-day high with weekly uptrend and volume
            long_breakout = (close[i] > high_20[i] and 
                           close[i] > ema_40_aligned[i] and
                           close[i] > ema_50[i] and
                           volume[i] > 1.5 * volume_ma_20_aligned[i])
            
            # Short: breakout below 20-day low with weekly downtrend and volume
            short_breakout = (close[i] < low_20[i] and 
                            close[i] < ema_40_aligned[i] and
                            close[i] < ema_50[i] and
                            volume[i] > 1.5 * volume_ma_20_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals