#!/usr/bin/env python3
"""
Experiment #5553: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 2.0x average and 
aligned with 12h Hull Moving Average (HMA) trend capture high-probability trend moves. 
HMA provides smooth trend filtering with reduced lag, while volume confirmation 
avoids false breakouts. Discrete position sizing (0.25) minimizes fee drag. 
Target: 19-50 trades/year (75-200 total over 4 years) with ATR-based trailing stoploss.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5553_4h_donchian20_12h_hma_vol_v1"
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
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        n_12h = len(close_12h)
        half_n = 12h_n // 2 if 12h_n >= 2 else 1
        sqrt_n = int(np.sqrt(n_12h))
        
        # WMA of full period
        wma_full = np.full(n_12h, np.nan)
        if n_12h >= 21:
            wma_full[20:] = wma(close_12h, 21)
        
        # WMA of half period
        wma_half = np.full(n_12h, np.nan)
        if n_12h >= half_n:
            wma_half[half_n-1:] = wma(close_12h, half_n)
        
        # Raw HMA: 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # WMA of raw HMA with sqrt(n) period
        hma_12h = np.full(n_12h, np.nan)
        if n_12h >= sqrt_n:
            wma_raw = wma(raw_hma, sqrt_n)
            hma_12h[sqrt_n-1:] = wma_raw
        
        # Align to LTF (4h) with shift(1) for completed bars only
        hma_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR price < HMA (trend reversal)
                if price <= stop_price or price <= donchian_low[i] or price < hma_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR price > HMA (trend reversal)
                if price >= stop_price or price >= donchian_high[i] or price > hma_aligned[i]:
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
        
        # Long: breakout above Donchian high with volume, above HMA (uptrend)
        long_entry = breakout_up and volume_confirmed and (price > hma_aligned[i])
        # Short: breakout below Donchian low with volume, below HMA (downtrend)
        short_entry = breakout_down and volume_confirmed and (price < hma_aligned[i])
        
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