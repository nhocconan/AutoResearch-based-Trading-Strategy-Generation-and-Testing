#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
Long when price breaks above Donchian upper band and close > 1d EMA50.
Short when price breaks below Donchian lower band and close < 1d EMA50.
Exit on opposite Donchian break or ATR trailing stop (highest/lowest since entry - 2.5*ATR).
Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
Donchian channels provide clear structure, 1d EMA50 filters medium-term trend,
ATR stoploss manages risk in volatile markets. Designed for both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # ATR(14) for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period) on primary timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        atr_val = atr[i]
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1d EMA50 (uptrend)
            if price > upper and price > ema50_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below Donchian lower AND price < 1d EMA50 (downtrend)
            elif price < lower and price < ema50_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
                # Exit long: price breaks below Donchian lower OR ATR trailing stop
                if price < lower or price < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price)
                # Exit short: price breaks above Donchian upper OR ATR trailing stop
                if price > upper or price > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_ATR_Trail"
timeframe = "4h"
leverage = 1.0