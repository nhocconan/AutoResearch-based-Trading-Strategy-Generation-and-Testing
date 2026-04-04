#!/usr/bin/env python3
"""
Experiment #3607: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts capture momentum bursts. Weekly pivot direction (from 1d data) filters for higher probability trades in the weekly trend. Volume confirmation ensures breakout strength. Works in bull markets (breakouts above weekly pivot in uptrend) and bear markets (breakdowns below weekly pivot in downtrend). Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3607_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's 1d data
    # Weekly high/low/close = max/min/last of prior 5 trading days (approx 1 week)
    lookback_week = 5
    weekly_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().values
    weekly_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().values
    weekly_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).apply(lambda x: x[-1]).values
    
    # Weekly pivot: P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed week only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # === 6h Indicators: Donchian Channel(20) for breakouts ===
    lookback_donchian = 20
    highest_high = pd.Series(high).rolling(window=lookback_donchian, min_periods=lookback_donchian).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_donchian, min_periods=lookback_donchian).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    
    warmup = max(lookback_donchian + 1, lookback_week + 1, 20, 14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Fixed stoploss: 2*ATR below entry for longs, above for shorts
            if position_side > 0:  # Long
                if price < stop_loss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > stop_loss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_ok = vol_ratio[i] > 1.5
        
        if volume_ok:
            # Determine bias from weekly pivot: price above pivot = bullish bias, below = bearish
            price_vs_pivot = price > weekly_pivot_aligned[i]
            
            # Long entry: price breaks above Donchian high AND above weekly pivot
            if (price > highest_high[i] and 
                price_vs_pivot):
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss_price = entry_price - 2.0 * atr[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND below weekly pivot
            elif (price < lowest_low[i] and 
                  not price_vs_pivot):
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss_price = entry_price + 2.0 * atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals