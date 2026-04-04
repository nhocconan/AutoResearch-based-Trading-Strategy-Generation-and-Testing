#!/usr/bin/env python3
"""
Experiment #5520: 4h Donchian(20) breakout + 1d HMA trend filter + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 1.3x average and aligned with 
1d HMA(21) trend capture sustained momentum moves while avoiding choppy markets. 
The 1d HMA provides a higher timeframe trend filter, reducing false breakouts in 
both bull and bear markets. Discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) 
control risk. Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5520_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for HMA trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        # Calculate HMA(21) on 1d data
        half_length = 21 // 2
        sqrt_length = int(np.sqrt(21))
        
        # WMA helper function
        def wma(values, period):
            if period <= 0:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1, dtype=np.float64)
            weights_sum = weights.sum()
            result = np.full_like(values, np.nan)
            for i in range(period - 1, len(values)):
                result[i] = np.dot(values[i - period + 1:i + 1], weights) / weights_sum
            return result
        
        # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(df_1d['close'].values, half_length)
        wma_full = wma(df_1d['close'].values, 21)
        raw_hma = 2 * wma_half - wma_full
        hma_1d = wma(raw_hma, sqrt_length)
        
        # Align to LTF (4h) with shift(1) for completed bars only
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
        # Uptrend: price > HMA, Downtrend: price < HMA
        uptrend = hma_1d_aligned > 0
        downtrend = hma_1d_aligned < 0
    else:
        hma_1d_aligned = np.full(n, np.nan)
        uptrend = np.full(n, False)
        downtrend = np.full(n, False)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume avg, ATR warmup
    
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
            np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Trend reversal: price < HMA (for long) or price > HMA (for short)
                if price <= stop_price or price <= donchian_low[i] or not uptrend[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Trend reversal: price > HMA (for short) or price < HMA (for long)
                if price >= stop_price or price >= donchian_high[i] or not downtrend[i]:
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
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirmed = volume_ratio[i] > 1.3
        
        # Entry conditions: breakout + volume + trend alignment
        if breakout_up and volume_confirmed and uptrend[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and downtrend[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals