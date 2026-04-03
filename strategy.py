#!/usr/bin/env python3
"""
Experiment #193: 4h Donchian Breakout + 12h HMA Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h with 12h HMA(21) trend filter and volume confirmation (>1.5x average) captures institutional breakouts in both bull and bear markets. Uses ATR(14) stoploss. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_193_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    def hma(series, period):
        """Hull Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan, dtype=np.float64)
        half = int(period / 2)
        sqrt = int(np.sqrt(period))
        wma_half = pd.Series(series).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_final = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean().values
        return hma_final
    
    close_12h = df_12h['close'].values
    hma_21_12h = hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Donchian(20) channels ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian low (contrarian exit)
                if price < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks above Donchian high (contrarian exit)
                if price > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume spike and 12h HMA uptrend
        if (price > donchian_high[i] and 
            close[i-1] <= donchian_high[i-1] and  # Was at or below Donchian high previous bar
            hma_21_12h_aligned[i] > close_12h[0] if i < len(close_12h) else hma_21_12h_aligned[i] > np.nanmean(close_12h) and  # Simplified trend check
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian low with volume spike and 12h HMA downtrend
        elif (price < donchian_low[i] and 
              close[i-1] >= donchian_low[i-1] and  # Was at or above Donchian low previous bar
              hma_21_12h_aligned[i] < close_12h[0] if i < len(close_12h) else hma_21_12h_aligned[i] < np.nanmean(close_12h) and  # Simplified trend check
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals