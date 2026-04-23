#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based position sizing.
Long when price breaks above Donchian upper band and close > 1d EMA50 (uptrend).
Short when price breaks below Donchian lower band and close < 1d EMA50 (downtrend).
Exit on opposite Donchian break or ATR trailing stop (2.5x ATR from extreme).
Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
Donchian channels provide objective breakout levels, 1d EMA50 filters medium-term trend.
Designed to capture strong momentum moves while avoiding whipsaws in both bull and bear markets.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss and position sizing
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        donchian_upper = high_ma[i]
        donchian_lower = low_ma[i]
        atr_val = atr[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1d EMA50 (uptrend)
            if price > donchian_upper and price > ema50_val:
                signals[i] = 0.30
                position = 1
                entry_price = price
                long_stop = price - 2.5 * atr_val
            # Short: price breaks below Donchian lower AND price < 1d EMA50 (downtrend)
            elif price < donchian_lower and price < ema50_val:
                signals[i] = -0.30
                position = -1
                entry_price = price
                short_stop = price + 2.5 * atr_val
        else:
            # Update trailing stops and check exit conditions
            if position == 1:
                # Update long trailing stop
                long_stop = max(long_stop, price - 2.5 * atr_val)
                # Exit long: price breaks below Donchian lower OR trailing stop hit
                if price < donchian_lower or price <= long_stop:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Update short trailing stop
                short_stop = min(short_stop, price + 2.5 * atr_val)
                # Exit short: price breaks above Donchian upper OR trailing stop hit
                if price > donchian_upper or price >= short_stop:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4H_Donchian20_1dEMA50_ATR_Trail"
timeframe = "4h"
leverage = 1.0