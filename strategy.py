#!/usr/bin/env python3
"""
Experiment #907: 6h Donchian(20) breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian breakouts on 6h capture momentum, filtered by weekly pivot bias (price above/below weekly pivot = bullish/bearish) and volume confirmation (>1.5x average). 
Weekly pivot derived from 1d data: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H. Long when price breaks above Donchian upper AND price > weekly pivot AND volume spike. 
Short when price breaks below Donchian lower AND price < weekly pivot AND volume spike. Uses discrete position sizing (0.25) to limit drawdown. 
Target: 75-150 total trades over 4 years (19-38/year). Works in bull (breakouts with trend) and bear (fades at extremes with pivot filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_907_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # Calculate weekly pivot levels from prior week's OHLC
    # We need to resample 1d to weekly properly - but since we can't resample,
    # we approximate by using the last 5 trading days' weekly OHLC
    # Simpler: use rolling weekly pivot based on last 5 days (1 week)
    lookback = 5  # 5 trading days = 1 week approx
    if len(high_1d) >= lookback:
        # Weekly high = max of last 5 daily highs
        weekly_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
        # Weekly low = min of last 5 daily lows
        weekly_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
        # Weekly close = last daily close in the week
        weekly_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).apply(lambda x: x[-1]).values
        # Weekly pivot = (H+L+C)/3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1 = 2*P - L
        weekly_r1 = 2 * weekly_pivot - weekly_low
        # Weekly S1 = 2*P - H
        weekly_s1 = 2 * weekly_pivot - weekly_high
    else:
        # Not enough data
        weekly_pivot = np.full(n, np.nan)
        weekly_r1 = np.full(n, np.nan)
        weekly_s1 = np.full(n, np.nan)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20)  # sufficient for Donchian, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i]) or
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
            
            # Optional: time-based exit after 8 bars (~2 days on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND price > weekly pivot (bullish bias)
            if price > upper_20[i] and price > weekly_pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND price < weekly pivot (bearish bias)
            elif price < lower_20[i] and price < weekly_pivot_aligned[i]:
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