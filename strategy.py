#!/usr/bin/env python3
"""
Experiment #5318: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: On daily timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 1.5x average and aligned with 1-week Hull Moving Average trend captures 
strong momentum moves in both bull and bear markets. Uses discrete position sizing (0.30) 
and ATR-based stoploss to control drawdown. Target: 10-25 trades/year on 1d timeframe 
(40-100 total over 4 years) to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5318_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(arr, weights_half, mode='valid') / weights_half.sum()
    wma_half = np.concatenate([np.full(half_period-1, np.nan), wma_half])
    
    # WMA for full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(arr, weights_full, mode='valid') / weights_full.sum()
    wma_full = np.concatenate([np.full(period-1, np.nan), wma_full])
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    wma_sqrt = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    hma = np.concatenate([np.full(sqrt_period-1, np.nan), wma_sqrt])
    
    return hma

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
    if len(df_1w) >= 21:
        hma_21 = calculate_hma(df_1w['close'].values, 21)
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Trade during active hours ---
        hour = hours[i]
        # Focus on major market hours: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
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
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below highest since entry
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price crosses below 1w HMA (trend reversal)
                if price <= stop_price or price <= donchian_low[i] or price < hma_21_aligned[i]:
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
                # 3. Price crosses above 1w HMA (trend reversal)
                if price >= stop_price or price >= donchian_high[i] or price > hma_21_aligned[i]:
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
        
        # HMA trend filter
        # Long: price above 1w HMA (uptrend)
        # Short: price below 1w HMA (downtrend)
        hma_long = price > hma_21_aligned[i-1]
        hma_short = price < hma_21_aligned[i-1]
        
        # Entry conditions: Donchian breakout + volume + HMA trend alignment
        if breakout_up and volume_confirmed and hma_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and hma_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals