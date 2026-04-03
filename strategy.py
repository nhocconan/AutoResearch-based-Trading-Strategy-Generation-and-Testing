#!/usr/bin/env python3
"""
Experiment #2067: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Weekly pivot points identify key institutional levels from the prior week. 
Combining Donchian breakouts with weekly pivot bias filters out false breakouts while 
capturing strong momentum moves. Volume confirmation ensures breakout validity. 
Works in bull/bear markets by using weekly structure for direction and 6h for precise timing.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2067_6h_donchian20_1w_pivot_vol_v1"
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
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from daily data
    # Week start: Monday 00:00 UTC (align with Binance weekly candles)
    # We'll resample daily to weekly using pandas (acceptable as preprocessing)
    df_1d_indexed = pd.DataFrame({
        'open': df_1d['open'],
        'high': df_1d['high'],
        'low': df_1d['low'],
        'close': df_1d['close']
    }, index=pd.to_datetime(df_1d['open_time']))
    
    # Resample to weekly (Monday start)
    weekly = df_1d_indexed.resample('W-MON', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly data to 1d index (forward fill from week start)
    weekly_index = weekly.index
    daily_index = pd.to_datetime(df_1d['open_time'])
    
    # Create aligned arrays for 1d timeframe
    pivot_1d = np.full(len(close_1d), np.nan)
    r1_1d = np.full(len(close_1d), np.nan)
    s1_1d = np.full(len(close_1d), np.nan)
    r2_1d = np.full(len(close_1d), np.nan)
    s2_1d = np.full(len(close_1d), np.nan)
    r3_1d = np.full(len(close_1d), np.nan)
    s3_1d = np.full(len(close_1d), np.nan)
    
    # Forward fill weekly values to daily
    for i, date in enumerate(daily_index):
        # Find most recent weekly bar (weekly index <= current date)
        mask = weekly_index <= date
        if np.any(mask):
            idx = np.where(mask)[0][-1]  # Last True index
            pivot_1d[i] = weekly_pivot[idx]
            r1_1d[i] = weekly_r1[idx]
            s1_1d[i] = weekly_s1[idx]
            r2_1d[i] = weekly_r2[idx]
            s2_1d[i] = weekly_s2[idx]
            r3_1d[i] = weekly_r3[idx]
            s3_1d[i] = weekly_s3[idx]
    
    # Determine weekly bias: price above pivot = bullish, below = bearish
    weekly_bias = np.where(close_1d > pivot_1d, 1, -1)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches weekly S1 (mean reversion to pivot support)
                elif price <= s1_1d[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches weekly R1 (mean reversion to pivot resistance)
                elif price >= r1_1d[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly pivot bias for direction filter
        bias = weekly_bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND weekly bias up
            if bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND weekly bias down
            elif bias < 0 and price < donchian_lower[i]:
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