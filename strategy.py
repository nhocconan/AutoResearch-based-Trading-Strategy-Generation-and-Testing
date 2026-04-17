#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR-based trailing stop.
Long when price breaks above 20-period Donchian high with above-average volume.
Short when price breaks below 20-period Donchian low with above-average volume.
Exit via ATR trailing stop (3*ATR from extreme) or opposite Donchian break.
Uses 12h EMA34 for trend filter to avoid counter-trend trades in choppy markets.
Target: 75-200 total trades over 4 years (19-50/year). Donchian breakouts capture strong momentum,
volume confirmation filters false breakouts, ATR stop manages risk, and 12h EMA reduces whipsaws.
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
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    stop_price = 0.0
    
    start_idx = max(50, lookback)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = volume_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and 12h uptrend
            if price > highest_high[i] and volume[i] > vol_ma and price > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                stop_price = price - 3.0 * atr_val
            # Short: price breaks below Donchian low with volume confirmation and 12h downtrend
            elif price < lowest_low[i] and volume[i] > vol_ma and price < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                stop_price = price + 3.0 * atr_val
        
        elif position == 1:
            # Update stop: trail up but never down
            stop_price = max(stop_price, price - 3.0 * atr_val)
            # Exit if stop hit or opposite Donchian break
            if price <= stop_price or price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update stop: trail down but never up
            stop_price = min(stop_price, price + 3.0 * atr_val)
            # Exit if stop hit or opposite Donchian break
            if price >= stop_price or price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeConfirm_12hEMA34_ATRStop"
timeframe = "4h"
leverage = 1.0