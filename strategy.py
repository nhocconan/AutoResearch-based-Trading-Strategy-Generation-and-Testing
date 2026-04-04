#!/usr/bin/env python3
"""
Experiment #5670: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: On daily timeframe, Donchian(20) breakouts with volume > 1.5x average and aligned 
with 1-week HMA(21) trend capture high-probability trend continuation moves. 
Weekly HMA provides structural trend filter that works in both bull and bear markets 
by avoiding entries against the higher timeframe trend. Volume confirms breakout strength. 
ATR trailing stop (2.5x) manages risk. Discrete sizing (0.25) minimizes fee churn. 
Target: 12-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5670_1d_donchian20_1w_hma_vol_v1"
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
    
    # === HTF: 1w data for HMA calculation ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights/weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        hma_raw = 2 * wma_half - wma_full
        hma_21 = wma(hma_raw, sqrt_len)
        
        # Pad to original length
        hma_21_padded = np.full(len(close_1w), np.nan)
        start_idx = 21 - 1  # WMA loses window-1 values
        hma_21_padded[start_idx:start_idx+len(hma_21)] = hma_21
        hma_21_values = hma_21_padded
    else:
        hma_21_values = np.full(len(df_1w), np.nan)
    
    # Align weekly HMA to daily timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_values)
    
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
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (UTC 21-23) ---
        hour = hours[i]
        if 21 <= hour <= 23:
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
                # Exit: stoploss OR price breaks below weekly HMA (trend change)
                if price <= stop_price or price <= hma_21_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above weekly HMA (trend change)
                if price >= stop_price or price >= hma_21_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Weekly HMA trend filter: long above HMA, short below HMA
        long_trend = price > hma_21_aligned[i]
        short_trend = price < hma_21_aligned[i]
        
        # Entry conditions: breakout in direction of weekly trend with volume
        long_setup = breakout_up and volume_confirmed and long_trend
        short_setup = breakout_down and volume_confirmed and short_trend
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals