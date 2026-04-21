#!/usr/bin/env python3
"""
4h_1d_Donchian20_VolumeRegime_ATRStop_V1
Hypothesis: Donchian(20) breakout with 1d EMA200 trend filter, volume confirmation, and ATR-based trailing stop.
Works in bull/bear: In uptrend (price>EMA200), long on upper band breakout; in downtrend (price<EMA200), short on lower band breakout.
Volume confirms breakout strength. ATR stop limits drawdown. Target: 25-40 trades/year per symbol (100-160 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Donchian channels on 4h data (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    volume_ok = prices['volume'].values > 1.5 * vol_ma
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 0:
            # Check for breakout with volume confirmation and trend filter
            bullish_breakout = (close_price > high_20[i]) and volume_ok[i] and (ema_200_1d_aligned[i] < close_price)
            bearish_breakout = (close_price < low_20[i]) and volume_ok[i] and (ema_200_1d_aligned[i] > close_price)
            
            if bullish_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = close_price
                atr_at_entry = atr[i]
            elif bearish_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = close_price
                atr_at_entry = atr[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price < entry - 2.5 * ATR (stoploss) or price < EMA200 (trend reversal)
            stop_price = entry_price - 2.5 * atr_at_entry
            trend_exit = close_price < ema_200_1d_aligned[i]
            
            if close_price <= stop_price or trend_exit:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price > entry + 2.5 * ATR (stoploss) or price > EMA200 (trend reversal)
            stop_price = entry_price + 2.5 * atr_at_entry
            trend_exit = close_price > ema_200_1d_aligned[i]
            
            if close_price >= stop_price or trend_exit:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_1d_Donchian20_VolumeRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0