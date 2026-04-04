#!/usr/bin/env python3
"""
Experiment #5587: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned 
with 1d pivot (R3/S3 for mean reversion, R4/S4 for breakout) capture high-probability moves.
In ranging markets (price between R3/S3), fade extremes. In trending markets (price outside R4/S4),
continuation breakouts work. Uses 1d pivots as dynamic support/resistance. ATR-based trailing 
stop (2.5x ATR) limits drawdown. Discrete position sizing (0.25) minimizes fee churn. 
Designed to work in bull (continuation breakouts) and bear (mean reversion at extremes).
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5587_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for pivot points ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate daily pivots: using previous day's OHLC
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        pivot_point = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2 * pivot_point - prev_low
        s1 = 2 * pivot_point - prev_high
        r2 = pivot_point + (prev_high - prev_low)
        s2 = pivot_point - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot_point - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot_point)
        r4 = prev_high + 3 * (pivot_point - prev_low)
        s4 = prev_low - 3 * (prev_high - pivot_point)
        
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        # Fallback if insufficient data
        pivot_aligned = r1_aligned = s1_aligned = r2_aligned = s2_aligned = \
                       r3_aligned = s3_aligned = r4_aligned = s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 2)  # Donchian, volume avg, ATR, pivot (need 2 days for prev day)
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR price reaches R4 (take profit)
                if price <= stop_price or price <= donchian_low[i] or price >= r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR price reaches S4 (take profit)
                if price >= stop_price or price >= donchian_high[i] or price <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Determine market regime based on pivot levels
        # Regime 1: Ranging (between S3 and R3) -> mean reversion at extremes
        # Regime 2: Trending (above R4 or below S4) -> continuation breakouts
        in_range = (price >= s3_aligned[i]) and (price <= r3_aligned[i])
        is_uptrend = price > r4_aligned[i]
        is_downtrend = price < s4_aligned[i]
        
        # Entry conditions:
        # In ranging market: fade at S3/R3 with volume
        # In trending market: breakout continuation with volume
        long_entry = False
        short_entry = False
        
        if in_range:
            # Ranging: mean reversion at S3/R3
            long_entry = (price <= s1_aligned[i] * 1.001) and volume_confirmed and (price > s3_aligned[i])
            short_entry = (price >= r1_aligned[i] * 0.999) and volume_confirmed and (price < r3_aligned[i])
        else:
            # Trending: continuation breakouts
            long_entry = breakout_up and volume_confirmed and is_uptrend
            short_entry = breakout_down and volume_confirmed and is_downtrend
        
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
    
    return signals