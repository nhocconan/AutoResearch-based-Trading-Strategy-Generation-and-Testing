#!/usr/bin/env python3
"""
Experiment #076: 12h Donchian Breakout + Volume + HTF Trend Filter
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume confirmation (>1.5x average)
and 1d timeframe trend filter (price above/below HMA(50)) capture sustained moves in both
bull and bear markets. The 1d HMA filter avoids counter-trend trades during regime shifts.
Target: 75-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_076_12h_donchian_vol_htf_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA(50) for trend filter ===
    def calculate_hma(arr, period):
        """Hull Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = arr[:-int(period/2):2] if len(arr) >= period//2 else arr
        wma2 = np.convolve(half, np.arange(1, len(half)+1), 'valid') / np.arange(1, len(half)+1).sum()
        wma1 = np.convolve(arr, np.arange(1, period+1), 'valid') / np.arange(1, period+1).sum()
        sqrt_period = int(np.sqrt(period))
        raw_hma = 2 * wma2 - wma1
        hma = np.convolve(raw_hma, np.arange(1, sqrt_period+1), 'valid') / np.arange(1, sqrt_period+1).sum()
        # Pad to original length
        result = np.full_like(arr, np.nan)
        start_idx = len(arr) - len(hma)
        result[start_idx:] = hma
        return result
    
    # Calculate HMA on 1d close
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h Indicators: Donchian Channels (20) ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian and volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_20_upper[i]) or np.isnan(donchian_20_lower[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # --- Trend Filter from 1d HMA ---
        # Long only if price > 1d HMA, Short only if price < 1d HMA
        long_allowed = price > hma_1d_aligned[i]
        short_allowed = price < hma_1d_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            if position_side > 0:  # Long
                # Exit on Donchian lower break or reverse signal
                if low[i] < donchian_20_lower[i] or (price < hma_1d_aligned[i] and vol_spike):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                # Exit on Donchian upper break or reverse signal
                if high[i] > donchian_20_upper[i] or (price > hma_1d_aligned[i] and vol_spike):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: Donchian upper break + volume + price > 1d HMA
        if (high[i] > donchian_20_upper[i-1] and vol_spike and long_allowed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian lower break + volume + price < 1d HMA
        elif (low[i] < donchian_20_lower[i-1] and vol_spike and short_allowed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals