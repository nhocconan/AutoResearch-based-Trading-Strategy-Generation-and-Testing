#!/usr/bin/env python3
"""
Experiment #697: 4h Donchian(20) breakout + 1d EMA(50) + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian(20) breakouts capture institutional breakouts with follow-through volume. 
Filtered by 1d EMA(50) trend direction and volume spike (>2x average). Uses ATR-based stoploss (2.0) 
and discrete position sizing (0.25) to minimize fee churn. Works in bull/bear markets via EMA trend filter.
Target: 75-200 total trades over 4 years (19-50/year). Uses 4h primary timeframe with 1d HTF for trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_697_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA(50) for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)  # auto shift(1) for completed 1d bars
    
    # === 4h Indicators: Donchian Channel(20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA and Donchian calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~24h on 4h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Get 1d EMA trend
            ema_trend = ema_1d_aligned[i]
            price_vs_ema = price > ema_trend
            
            # Long: price breaks above Donchian upper + price > EMA(50)
            if price > donchian_upper[i] and price_vs_ema:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower + price < EMA(50)
            elif price < donchian_lower[i] and not price_vs_ema:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals