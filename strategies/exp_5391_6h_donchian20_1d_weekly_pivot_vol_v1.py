#!/usr/bin/env python3
"""
Experiment #5391: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 2.0x average and aligned with the 1d weekly pivot (price above/below weekly pivot) 
captures strong momentum moves while minimizing false breakouts. The weekly pivot acts as 
a higher timeframe directional filter to avoid counter-trend trades. Discrete position sizing 
(0.25) and ATR-based trailing stoploss (2.5x ATR) control risk. Target: 12-37 trades/year 
(50-150 total over 4 years) to minimize fee drag while maintaining statistical significance. 
Works in bull markets via breakouts above rising weekly pivot and in bear markets via short 
breakdowns below falling weekly pivot. Using 1d HTF instead of 1w to increase signal frequency 
while keeping trade count manageable.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5391_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from 1d OHLC (using prior week's data)
        # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        # We'll use rolling window of 5 days (1 week) to approximate
        high_1d = pd.Series(df_1d['high'].values)
        low_1d = pd.Series(df_1d['low'].values)
        close_1d = pd.Series(df_1d['close'].values)
        
        # Weekly high/low/close over 5-day window
        weekly_high = high_1d.rolling(window=5, min_periods=5).max().values
        weekly_low = low_1d.rolling(window=5, min_periods=5).min().values
        weekly_close = close_1d.rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot point
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Align to LTF (6h) with shift(1) for completed bars only
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot) if len(weekly_pivot) > 0 else np.full(n, np.nan)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
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
    
    warmup = max(20, 20, 14, 5)  # Donchian, volume avg, ATR, 1d weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below highest since entry
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price crosses below weekly pivot (trend reversal)
                if price <= stop_price or price <= donchian_low[i] or price < weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.5 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price crosses above weekly pivot (trend reversal)
                if price >= stop_price or price >= donchian_high[i] or price > weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 2.0x average volume (stricter than 1.8x)
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Weekly pivot direction filter
        # Long: price above weekly pivot (bullish bias)
        # Short: price below weekly pivot (bearish bias)
        pivot_bias_up = price > weekly_pivot_aligned[i-1]
        pivot_bias_down = price < weekly_pivot_aligned[i-1]
        
        # Entry conditions
        if breakout_up and volume_confirmed and pivot_bias_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and pivot_bias_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals