#!/usr/bin/env python3
"""
Experiment #1871: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts capture strong momentum. Weekly pivot (from 1d data) provides institutional reference levels: price above weekly pivot = bullish bias, below = bearish bias. Volume confirmation (>1.5x average) ensures breakouts have conviction. Works in both bull and bear markets by aligning with 1d trend via weekly pivot. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1871_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly pivot from prior week (using 1d data: (PriorWeek High + Low + Close) / 3)
    # Need to compute weekly OHLC from daily data
    # We'll approximate: for each day, use prior 5-day high/low/close
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 1d EMA(50) for additional trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Donchian(20) channels ===
    # Donchian Upper = highest high of past 20 periods
    # Donchian Lower = lowest low of past 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or time-based ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian Lower (failed breakout)
                if price < donchian_low[i]:
                    exit_signal = True
                # Exit if 1d trend flips against position
                elif trend_1d_aligned[i] < 0:
                    exit_signal = True
                # Time-based exit: max 10 bars (~60 hours) to avoid overstaying
                elif bars_since_entry >= 10:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian Upper (failed breakdown)
                if price > donchian_high[i]:
                    exit_signal = True
                # Exit if 1d trend flips against position
                elif trend_1d_aligned[i] > 0:
                    exit_signal = True
                # Time-based exit
                elif bars_since_entry >= 10:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_confirmed = vol_ratio[i] > 1.5
        
        # Require alignment with 1d trend (via EMA)
        trend_aligned = trend_1d_aligned[i]
        
        if volume_confirmed:
            # Long: price breaks above Donchian Upper AND above weekly pivot AND 1d trend up
            if (price > donchian_high[i] and 
                price > weekly_pivot_aligned[i] and 
                trend_aligned > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian Lower AND below weekly pivot AND 1d trend down
            elif (price < donchian_low[i] and 
                  price < weekly_pivot_aligned[i] and 
                  trend_aligned < 0):
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

}