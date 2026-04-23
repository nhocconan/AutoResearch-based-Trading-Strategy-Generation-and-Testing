#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR(14) stoploss.
Long when price breaks above Donchian upper band AND close > 12h EMA50.
Short when price breaks below Donchian lower band AND close < 12h EMA50.
Exit via ATR trailing stop: signal→0 when price < highest_high_since_entry - 2.5*ATR (long) 
or price > lowest_low_since_entry + 2.5*ATR (short).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Donchian channels provide clear structure with low false breakouts in ranging markets.
12h EMA50 offers smoother trend filter than shorter EMAs, reducing whipsaw in bear markets.
ATR stoploss adapts to volatility, protecting gains during sharp reversals.
Designed to work in both bull and bear markets by using HTF trend filter and volatility-adjusted exits.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 14)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 12h EMA50
            if price > highest_high[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below Donchian lower band AND close < 12h EMA50
            elif price < lowest_low[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_ATRStop"
timeframe = "4h"
leverage = 1.0