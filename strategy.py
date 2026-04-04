#!/usr/bin/env python3
"""
Experiment #5553: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 1.8x average and 
aligned with 12h Hull Moving Average (HMA) trend capture high-probability trend moves. 
The 12h HMA provides smooth trend filtering with reduced lag, while volume confirmation 
filters false breakouts. ATR-based trailing stop manages risk. Target: 19-50 trades/year 
(75-200 total over 4 years) with discrete position sizing (0.25) to minimize fee drag.
Works in bull markets via breakout continuation and bear markets via short breakdowns.
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
    
    # === HTF: 12h data for HMA trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Calculate Hull Moving Average (HMA) on 12h close
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_12h = df_12h['close'].values
        n_12h = len(close_12h)
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        if n_12h >= 21:
            wma_full = np.array([np.nan] * n_12h)
            wma_half = np.array([np.nan] * n_12h)
            
            for i in range(half_n, n_12h):
                wma_half[i] = wma(close_12h[i-half_n+1:i+1], half_n)
            
            for i in range(21, n_12h):
                wma_full[i] = wma(close_12h[i-21+1:i+1], 21)
            
            # HMA = WMA(2*WMA(half) - WMA(full), sqrt(n))
            hma_12h = np.array([np.nan] * n_12h)
            for i in range(21, n_12h):
                if not np.isnan(wma_full[i]) and not np.isnan(wma_half[i]):
                    diff = 2 * wma_half[i] - wma_full[i]
                    if i >= half_n + sqrt_n - 1:
                        hma_12h[i] = wma(diff[max(0, i-half_n-sqrt_n+2):i+1], sqrt_n)[-1]
            
            # Align to LTF (4h) with shift(1) for completed bars only
            hma_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
        else:
            hma_aligned = np.full(n, np.nan)
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
            np.isnan(hma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR price < HMA (trend change)
                if price <= stop_price or price <= donchian_low[i] or price < hma_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR price > HMA (trend change)
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
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Long: breakout above Donchian high with volume and price > HMA (uptrend)
        long_entry = breakout_up and volume_confirmed and (price > hma_aligned[i])
        # Short: breakout below Donchian low with volume and price < HMA (downtrend)
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