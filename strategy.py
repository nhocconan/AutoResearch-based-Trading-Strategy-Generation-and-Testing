#!/usr/bin/env python3
"""
Experiment #6073: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 12h Hull Moving Average capture medium-term swings with noise filtering. 12h HMA provides smoother trend direction than EMA/SMA, reducing whipsaws in sideways markets. Volume >1.5x average confirms institutional participation. ATR-based trailing stop manages risk. Designed for 75-200 trades over 4 years (19-50/year) with discrete sizing (0.25) to minimize fee drag. Works in bull markets (breakouts above rising 12h HMA) and bear markets (breakdowns below falling 12h HMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6073_4h_donchian20_12h_hma_vol_v1"
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
    
    # === HTF: 12h data for HMA(21) trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Calculate Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 12 // 2  # 6 for 12-period
        sqrt_len = int(np.sqrt(12))  # 3 for 12-period
        
        # WMA helper function
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMA for full period
        wma_full = np.array([wma(close[max(0, i-12+1):i+1], 12) if i >= 11 else np.nan for i in range(len(close))])
        # Calculate WMA for half period
        wma_half = np.array([wma(close[max(0, i-6+1):i+1], 6) if i >= 5 else np.nan for i in range(len(close))])
        # HMA = WMA(2*WMA(6) - WMA(12), 3)
        raw_hma = 2 * wma_half - wma_full
        hma_values = np.array([wma(raw_hma[max(0, i-3+1):i+1], 3) if i >= 2 else np.nan for i in range(len(raw_hma))])
        # Align to 4h timeframe with shift(1) for completed bars only
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_values)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 20, 14) + 1  # Donchian, volume avg, HMA calc, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:59 UTC) ---
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
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
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
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter for stronger signals
        
        # Multi-timeframe trend filter: price must be aligned with 12h HMA
        bullish_trend = price > hma_12h_aligned[i]
        bearish_trend = price < hma_12h_aligned[i]
        
        # Entry conditions:
        # Long: breakout up with volume AND bullish trend on 12h HMA
        # Short: breakout down with volume AND bearish trend on 12h HMA
        long_entry = breakout_up and volume_confirmed and bullish_trend
        short_entry = breakout_down and volume_confirmed and bearish_trend
        
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