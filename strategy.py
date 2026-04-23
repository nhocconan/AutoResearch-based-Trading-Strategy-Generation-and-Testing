#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend + volume confirmation + ATR stoploss.
Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when price crosses the opposite Donchian band (middle) or ATR stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Designed for BTC/ETH resilience in both bull and bear markets via trend filter and volatility-adjusted stops.
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
    
    # Load 12h data for Donchian channels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2.0
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 12h data for stoploss
    tr1 = np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1)))
    tr2 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_12h[0] - low_12h[0]  # first bar
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Align 12h indicators to 12h timeframe (identity, but required for consistency)
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_12h, middle_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or np.isnan(middle_20_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 1d EMA34 AND volume spike
            if (price > highest_20_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND close < 1d EMA34 AND volume spike
            elif (price < lowest_20_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian middle or ATR stoploss
                if price < middle_20_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian middle or ATR stoploss
                if price > middle_20_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA34_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0