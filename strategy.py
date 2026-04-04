#!/usr/bin/env python3
"""
Experiment #6264: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA(21) trend capture institutional order flow. 
Volume >1.5x average confirms participation. Uses 1w HTF for HMA trend (proven effective for identifying key trend direction). 
Discrete sizing (0.25) manages fee drag. Target: 30-100 trades over 4 years (7-25/year) for 1d timeframe. 
Works in both bull (breakout continuation with trend) and bear (mean reversion at extremes against trend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6264_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for HMA trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:  # Need at least 21 weekly bars for HMA
        # Calculate HMA(21) on weekly close
        close_1w = df_1w['close'].values
        n_1w = len(close_1w)
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        # WMA helper
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(values, weights[::-1], mode='valid') / weights.sum()
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(close_1w, half_n)
        wma_full = wma(close_1w, 21)
        # Align arrays: WMA(half) starts at index half_n-1, WMA(full) at index 20
        # We need same length, so pad appropriately
        hma_1w = np.full(n_1w, np.nan)
        if len(wma_half) >= half_n and len(wma_full) >= 21:
            # 2*WMA(half) - WMA(full) aligned to start at index 20 (for WMA_full)
            diff = 2 * wma_half[half_n-1:] - wma_full
            # Take last sqrt_n elements of diff for final WMA
            if len(diff) >= sqrt_n:
                hma_1w[20+sqrt_n-1:] = wma(diff[-sqrt_n:], sqrt_n)
        
        # Align to 1d timeframe (shift(1) inside align_htf_to_ltf for completed bars only)
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 1d Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 21) + 1  # Donchian, volume avg, ATR, HMA warmup + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Mean reversion: price reaches Donchian midpoint in strong trend
                midpoint = (donchian_high[i] + donchian_low[i]) / 2
                if price <= stop_price or price <= donchian_low[i] or (hma_1w_aligned[i] > price and price <= midpoint):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Mean reversion: price reaches Donchian midpoint in strong trend
                midpoint = (donchian_high[i] + donchian_low[i]) / 2
                if price >= stop_price or price >= donchian_high[i] or (hma_1w_aligned[i] < price and price >= midpoint):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter
        
        # Entry logic based on HMA trend:
        # LONG: Breakout above Donchian high with volume AND price > HMA (bullish trend)
        # SHORT: Breakout below Donchian low with volume AND price < HMA (bearish trend)
        long_entry = breakout_up and volume_confirmed and price > hma_1w_aligned[i]
        short_entry = breakout_down and volume_confirmed and price < hma_1w_aligned[i]
        
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