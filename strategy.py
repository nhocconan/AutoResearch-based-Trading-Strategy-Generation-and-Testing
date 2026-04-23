#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR trailing stop.
Long when price breaks above Donchian(20) high AND close > 1d EMA34 AND volume > 1.5x average.
Short when price breaks below Donchian(20) low AND close < 1d EMA34 AND volume > 1.5x average.
Exit when price touches opposite Donchian(10) level or ATR stoploss hit.
Uses discrete sizing (0.30) to balance return and drawdown. Targets 25-40 trades/year per symbol.
Donchian channels provide objective structure, EMA34 filters trend direction, volume confirms breakout strength.
Works in bull (breakouts continue) and bear (breakdowns continue) markets when aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h data
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian(10) for exit
    highest_high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    
    # Align 4h indicators to LTF
    highest_high_20_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    highest_high_10_aligned = align_htf_to_ltf(prices, df_4h, highest_high_10)
    lowest_low_10_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_10)
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to LTF
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_bar = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 4h close for price comparison (aligned to LTF)
        price_4h = close_4h[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Donchian(20) breakout + trend filter + volume confirmation
            if (price_4h > highest_high_20_aligned[i] and 
                price_4h > ema34_1d_aligned[i] and 
                volume_1d[i] > 1.5 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price_4h
                entry_bar = i
            # Short: Donchian(20) breakdown + trend filter + volume confirmation
            elif (price_4h < lowest_low_20_aligned[i] and 
                  price_4h < ema34_1d_aligned[i] and 
                  volume_1d[i] > 1.5 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price_4h
                entry_bar = i
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Donchian(10) low OR ATR stoploss
                if price_4h <= lowest_low_10_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss (2.5 * ATR)
                elif price_4h < entry_price - 2.5 * atr_4h_aligned[i]:
                    exit_signal = True
                # Time-based exit: max 10 bars (approx 40 hours on 4h)
                elif i - entry_bar >= 10:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches Donchian(10) high OR ATR stoploss
                if price_4h >= highest_high_10_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss (2.5 * ATR)
                elif price_4h > entry_price + 2.5 * atr_4h_aligned[i]:
                    exit_signal = True
                # Time-based exit: max 10 bars
                elif i - entry_bar >= 10:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_ATRStop_TimeExit"
timeframe = "4h"
leverage = 1.0