#!/usr/bin/env python3
"""
Experiment #5784: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
HYPOTHESIS: 1d Donchian breakouts aligned with 1w HMA(21) trend capture strong continuation moves with volume confirmation. Uses 1w timeframe for structure to reduce whipsaws, targeting 75-200 trades over 4 years. Works in bull/bear markets by requiring breakout alignment with higher timeframe trend direction. Discrete sizing 0.25 minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5784_1d_donchian20_1w_hma_vol_v1"
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
    
    # === HTF: 1w data for HMA(21) trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        close_1w = df_1w['close'].values
        half_length = 21 // 2
        sqrt_length = int(np.sqrt(21))
        
        # WMA function
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(close_1w, half_length)
        wma_full = wma(close_1w, 21)
        # Pad to align lengths
        wma_half_padded = np.full_like(close_1w, np.nan)
        wma_half_padded[half_length-1:] = wma_half
        wma_full_padded = np.full_like(close_1w, np.nan)
        wma_full_padded[20:] = wma_full
        
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_1w = wma(raw_hma, sqrt_length)
        # Pad HMA result
        hma_1w_padded = np.full_like(close_1w, np.nan)
        hma_1w_padded[sqrt_length-1:] = hma_1w[:len(close_1w)-sqrt_length+1]
    else:
        hma_1w_padded = np.full(len(df_1w), np.nan)
    
    # Align 1w HMA to 1d timeframe (shifted by 1 for completed 1w bars only)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
    
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
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
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
                # Exit: stoploss OR price breaks below Donchian low (failed breakout) OR trend reversal
                if price <= stop_price or price <= donchian_low[i] or close[i] < hma_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout) OR trend reversal
                if price >= stop_price or price >= donchian_high[i] or close[i] > hma_1w_aligned[i]:
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
        # Trend alignment: price above/below HMA for trend confirmation
        trend_up = close[i] > hma_1w_aligned[i]
        trend_down = close[i] < hma_1w_aligned[i]
        
        # Entry conditions: breakout in direction of higher timeframe trend with volume confirmation
        long_setup = breakout_up and trend_up and volume_confirmed
        short_setup = breakout_down and trend_down and volume_confirmed
        
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
</truncated>