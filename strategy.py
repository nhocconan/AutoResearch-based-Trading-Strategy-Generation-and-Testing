#!/usr/bin/env python3
"""
Experiment #5533: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 1.5x average and aligned with 
12h HMA(21) trend capture high-probability moves in both bull and bear markets. The 12h HMA 
provides smooth trend filtering that works across regimes, while volume confirmation filters 
false breakouts. Discrete position sizing (0.25) and ATR-based trailing stop control risk. 
Target: 19-50 trades/year (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5533_4h_donchian20_12h_hma_vol_v1"
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
    
    # === HTF: 12h data for HMA(21) trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Calculate HMA(21) on 12h close
        close_12h = df_12h['close'].values
        n_12h = len(close_12h)
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        # WMA function for HMA calculation
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(close_12h, half_n)
        wma_full = wma(close_12h, 21)
        # Handle edge cases for WMA arrays
        wma_2x_sub = np.full_like(close_12h, np.nan)
        wma_2x_sub[half_n-1:] = 2 * wma_half[:len(wma_2x_sub[half_n-1:])]
        wma_diff = wma_2x_sub - wma_full
        hma_12h = wma(wma_diff[~np.isnan(wma_diff)], sqrt_n) if np.sum(~np.isnan(wma_diff)) >= sqrt_n else np.array([])
        # Pad HMA array to match original length
        hma_12h_full = np.full(n_12h, np.nan)
        if len(hma_12h) > 0:
            start_idx = 21 - 1  # WMA(21) loses 20 points, WMA(sqrt) loses sqrt_n-1 more
            end_idx = start_idx + len(hma_12h)
            if end_idx <= n_12h:
                hma_12h_full[start_idx:end_idx] = hma_12h
        # Align to LTF (4h) with shift(1) for completed bars only
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_full)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    highest_since_entry = 0.0  # For long positions
    lowest_since_entry = 0.0   # For short positions
    
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
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend failure ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry (trailing stop)
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price closes below 12h HMA (trend failure)
                if price <= stop_price or price <= donchian_low[i] or price < hma_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry (trailing stop)
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price closes above 12h HMA (trend failure)
                if price >= stop_price or price >= donchian_high[i] or price > hma_12h_aligned[i]:
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
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume_ratio[i] > 1.5
        
        # 12h HMA trend filter
        hma_trend_up = hma_12h_aligned[i] > hma_12h_aligned[i-1]  # HMA rising
        hma_trend_down = hma_12h_aligned[i] < hma_12h_aligned[i-1]  # HMA falling
        
        # Entry conditions: breakout + volume + trend alignment
        if breakout_up and volume_confirmed and hma_trend_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and hma_trend_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals