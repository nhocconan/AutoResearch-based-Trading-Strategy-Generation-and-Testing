#!/usr/bin/env python3
"""
Experiment #5271: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with 1d weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) and volume confirmation (volume > 1.5x 20-period average) capture strong momentum moves with filtered false breakouts. Weekly pivots provide structural support/resistance from higher timeframe, reducing whipsaws. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets by buying breakouts above weekly pivot and in bear markets by selling breakdowns below weekly pivot, avoiding range-bound conditions where price oscillates around pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5271_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC (using last 5 trading days approximation)
        # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        df_1d['weekly_high'] = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
        df_1d['weekly_low'] = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
        df_1d['weekly_close'] = df_1d['close'].shift(5)  # Close 5 days ago (prior week)
        df_1d['weekly_pivot'] = (df_1d['weekly_high'] + df_1d['weekly_low'] + df_1d['weekly_close']) / 3.0
        weekly_pivot = df_1d['weekly_pivot'].values
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Current volume vs 20-period average
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 20)  # Donchian, volume MA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (6h timeframe, full day) ---
        # No session filter - 6h candles already distributed throughout day
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_ratio[i] > 1.5  # Volume > 1.5x 20-period average
        
        # --- Exit Logic: Close position when breakout fails or pivot reverses ---
        if in_position:
            # Check for breakout failure (price re-enters Donchian channel)
            breakout_long = price > donchian_high[i]
            breakout_short = price < donchian_low[i]
            
            # Check pivot alignment
            pivot_bullish = price > weekly_pivot_aligned[i]
            pivot_bearish = price < weekly_pivot_aligned[i]
            
            # Exit conditions:
            # 1. Breakout fails (price re-enters channel)
            # 2. Pivot direction reverses against position
            if position_side > 0:  # Long position
                if (not breakout_long) or (not pivot_bullish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if (not breakout_short) or (not pivot_bearish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i]
        breakout_short = price < donchian_low[i]
        
        # Pivot direction from 1d weekly pivot
        pivot_bullish = price > weekly_pivot_aligned[i]
        pivot_bearish = price < weekly_pivot_aligned[i]
        
        # Entry conditions: Donchian breakout + pivot alignment + volume confirmation
        if breakout_long and pivot_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_short and pivot_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals