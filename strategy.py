#!/usr/bin/env python3
"""
Experiment #6270: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA trend capture institutional momentum. Volume >2.0x average confirms participation. Uses 1d timeframe with 1w HTF for trend filter. Discrete sizing (0.25) manages fee drag. Target: 30-100 trades over 4 years (7-25/year) for 1d timeframe. Works in bull markets via breakout continuation and bear markets via mean reversion at extremes when price reaches opposite Donchian band after extreme move.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6270_1d_donchian20_1w_hma_vol_v1"
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
    
    # === HTF: 1w data for HMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:  # Need enough for HMA calculation
        # Calculate HMA(21) on weekly close
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        # Calculate WMA for half period
        wma_half = np.array([np.nan] * len(close_1w))
        for i in range(half_len, len(close_1w)):
            wma_half[i] = wma(close_1w[i-half_len+1:i+1], half_len)
        
        # Calculate WMA for full period
        wma_full = np.array([np.nan] * len(close_1w))
        for i in range(21, len(close_1w)):
            wma_full[i] = wma(close_1w[i-21+1:i+1], 21)
        
        # HMA = WMA(2*WMA_half - WMA_full, sqrt_len)
        hma_raw = 2 * wma_half - wma_full
        hma_21 = np.array([np.nan] * len(close_1w))
        for i in range(int(sqrt_len), len(hma_raw)):
            if not np.isnan(hma_raw[i]):
                hma_21[i] = wma(hma_raw[i-int(sqrt_len)+1:i+1], sqrt_len)
        
        # Align to 1d timeframe (shift(1) inside align_htf_to_ltf for completed bars only)
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 21) + 1  # Donchian, volume avg, ATR, HMA + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_21_aligned[i])):
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
                # 3. Mean reversion: price reaches opposite Donchian band after strong move
                if price <= stop_price or price <= donchian_low[i] or price <= donchian_low[i] + (donchian_high[i-20] - donchian_low[i-20]) * 0.5:
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
                # 3. Mean reversion: price reaches opposite Donchian band after strong move
                if price >= stop_price or price >= donchian_high[i] or price >= donchian_high[i] - (donchian_high[i-20] - donchian_low[i-20]) * 0.5:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Strong volume filter
        
        # Trend filter: HMA slope (using 3-period change)
        hma_slope = hma_21_aligned[i] - hma_21_aligned[i-3] if i >= 3 else 0
        
        # Entry logic:
        # LONG: Breakout above Donchian high with volume AND weekly HMA trending up
        # SHORT: Breakout below Donchian low with volume AND weekly HMA trending down
        long_entry = breakout_up and volume_confirmed and hma_slope > 0
        short_entry = breakout_down and volume_confirmed and hma_slope < 0
        
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