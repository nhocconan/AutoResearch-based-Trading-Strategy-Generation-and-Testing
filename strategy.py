#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss.
Long when price breaks above 20-day high AND 1w EMA50 is rising.
Short when price breaks below 20-day low AND 1w EMA50 is falling.
Exit via ATR trailing stop (3x ATR) or opposite Donchian breakout.
Uses 1w HTF for EMA50 trend filter to avoid counter-trend whipsaws in bear markets.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Donchian channels provide clear structure; EMA50 on weekly filters trend regime.
ATR stoploss manages risk during volatile periods.
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
    
    # Calculate 20-period Donchian channels (primary timeframe)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 14, 50)  # Donchian(20), ATR(14), EMA50(1w)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        price = close[i]
        highest = highest_high[i]
        lowest = lowest_low[i]
        atr_val = atr[i]
        ema_val = ema_50_aligned[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above 20-day high AND 1w EMA50 rising
            if price > highest and ema_rising:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Break below 20-day low AND 1w EMA50 falling
            elif price < lowest and ema_falling:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
                # Exit conditions: ATR trailing stop OR price breaks below 20-day low
                if price <= highest_since_entry - 3.0 * atr_val or price < lowest:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
                # Exit conditions: ATR trailing stop OR price breaks above 20-day high
                if price >= lowest_since_entry + 3.0 * atr_val or price > highest:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_ATRStop"
timeframe = "1d"
leverage = 1.0