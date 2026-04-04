#!/usr/bin/env python3
"""
Experiment #4035: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts aligned with weekly pivot bias (bullish when price > weekly pivot, bearish when price < weekly pivot) 
and volume confirmation (>2.0x MA20) capture high-probability continuation moves in both bull and bear markets. 
Weekly pivot provides structural support/resistance from higher timeframe, reducing whipsaws. ATR(20) trailing stop (2.5x) 
controls drawdown. Discrete sizing (0.25) limits fee churn. Target: 75-200 total trades over 4 years (19-50/year).
Works in bull/bear: In bull markets, buy upper breakouts above weekly pivot; in bear markets, sell lower breakouts below weekly pivot. 
Volume filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4035_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w pivot levels (using prior week's OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly pivot: P = (H + L + C) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Trend filter: price relative to weekly pivot
            price_above_pivot = price > weekly_pivot_aligned[i]
            price_below_pivot = price < weekly_pivot_aligned[i]
            
            # Breakout logic: 
            # - Bullish bias (price > weekly pivot): look for long on upper Donchian breakout
            # - Bearish bias (price < weekly pivot): look for short on lower Donchian breakout
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long conditions: bullish bias + upper Donchian breakout
            long_entry = breakout_up and price_above_pivot
            
            # Short conditions: bearish bias + lower Donchian breakout
            short_entry = breakout_down and price_below_pivot
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals