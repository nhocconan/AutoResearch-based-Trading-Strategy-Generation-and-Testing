#!/usr/bin/env python3
"""
Experiment #5990: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Donchian breakouts on 1d aligned with 1-week HMA trend (price above/below HMA = bullish/bearish)
capture sustained moves with lower noise. Weekly HMA provides structural bias more resilient to daily noise.
Volume >1.5x average confirms breakout strength. ATR trailing stop manages risk. Target 30-100 trades over 4 years.
Works in both bull/bear: weekly HMA trend prevents counter-trend entries, volume confirmation avoids false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5990_1d_donchian20_1w_hma_vol_v1"
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
    if len(df_1w) >= 21:  # Need at least 21 periods for HMA
        # Calculate HMA(21) on weekly close
        hma_21 = calculate_hma(df_1w['close'].values, 21)
        # Align to 1d timeframe with shift(1) for completed weekly bars only
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
    
    warmup = max(20, 20, 14, 21) + 1  # Donchian, volume avg, ATR, HMA lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
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
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
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
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Weekly HMA trend bias: price above/below weekly HMA
        above_hma = price > hma_21_aligned[i]
        below_hma = price < hma_21_aligned[i]
        
        # Entry conditions: 
        # Long: breakout up with volume AND above weekly HMA
        # Short: breakout down with volume AND below weekly HMA
        long_setup = breakout_up and volume_confirmed and above_hma
        short_setup = breakout_down and volume_confirmed and below_hma
        
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

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(values)
    for i in range(half_period, len(values)):
        wma_half[i] = np.nansum(values[i-half_period+1:i+1] * np.arange(1, half_period+1)) / (half_period * (half_period + 1) / 2)
    
    # WMA of full period
    wma_full = np.zeros_like(values)
    for i in range(period, len(values)):
        wma_full[i] = np.nansum(values[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
    
    # Raw HMA: 2 * WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = np.zeros_like(values)
    for i in range(sqrt_period, len(values)):
        hma[i] = np.nansum(raw_hma[i-sqrt_period+1:i+1] * np.arange(1, sqrt_period+1)) / (sqrt_period * (sqrt_period + 1) / 2)
    
    return hma