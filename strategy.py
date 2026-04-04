#!/usr/bin/env python3
"""
Experiment #5673: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned 
with 12h HMA(21) trend direction capture high-probability trend continuation moves. 
The 12h HMA provides a smoother trend filter that works in both bull and bear markets 
by avoiding counter-trend entries. Volume confirms breakout strength. ATR trailing stop 
(2.0x) manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 19-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5673_4h_donchian20_12h_hma_vol_v1"
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
        n_hma = 21
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_n = n_hma // 2
        sqrt_n = int(np.sqrt(n_hma))
        
        def wma(arr, window):
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        # Calculate WMA for half period
        if len(close_12h) >= half_n:
            wma_half = wma(close_12h, half_n)
            wma_half = np.concatenate([np.full(half_n-1, np.nan), wma_half])
        else:
            wma_half = np.full(len(close_12h), np.nan)
        
        # Calculate WMA for full period
        if len(close_12h) >= n_hma:
            wma_full = wma(close_12h, n_hma)
            wma_full = np.concatenate([np.full(n_hma-1, np.nan), wma_full])
        else:
            wma_full = np.full(len(close_12h), np.nan)
        
        # Raw HMA: 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # Final HMA: WMA(raw_hma, sqrt_n)
        if len(raw_hma) >= sqrt_n:
            hma_values = wma(raw_hma[~np.isnan(raw_hma)], sqrt_n)
            # Reconstruct with NaNs
            hma_12h = np.full(len(raw_hma), np.nan)
            valid_idx = ~np.isnan(raw_hma)
            if np.sum(valid_idx) >= sqrt_n:
                hma_12h[valid_idx] = np.concatenate([np.full(sqrt_n-1, np.nan), hma_values])
        else:
            hma_12h = np.full(len(close_12h), np.nan)
    else:
        hma_12h = np.full(len(df_12h), np.nan)
    
    # Align 12h HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
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
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
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
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (trend reversal)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (trend reversal)
                if price >= stop_price or price >= donchian_high[i]:
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
        
        # 12h HMA trend filter: long when price > HMA, short when price < HMA
        long_trend = price > hma_12h_aligned[i]
        short_trend = price < hma_12h_aligned[i]
        
        # Entry conditions: breakout in direction of 12h trend with volume confirmation
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

</think>