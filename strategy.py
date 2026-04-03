#!/usr/bin/env python3
"""
Experiment #278: 1d Donchian Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: Daily Donchian(20) breakouts aligned with weekly HMA(21) trend direction, confirmed by volume spikes (>1.5x 20-day average), capture strong momentum moves in both bull and bear markets. The weekly HMA filter ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts. Volume confirmation adds institutional participation validation. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to minimize fee drag while capturing high-probability trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            raw_hma = 2 * wma_half - wma_full
            hma_21_1w = wma(raw_hma, sqrt_len)
            # Pad to original length
            hma_21_1w_padded = np.full(len(close_1w), np.nan)
            hma_21_1w_padded[half_len:half_len+len(hma_21_1w)] = hma_21_1w
            hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
        else:
            hma_21_1w_aligned = np.full(n, np.nan)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    # Volume Spike: >1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Breakout Conditions ---
        breakout_high = close[i] > donchian_high[i]
        breakout_low = close[i] < donchian_low[i]
        
        # --- Weekly HMA Trend Filter ---
        price_above_hma = close[i] > hma_21_1w_aligned[i]
        price_below_hma = close[i] < hma_21_1w_aligned[i]
        
        # --- Volume Confirmation ---
        vol_spike = volume_spike[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian High + price above weekly HMA + volume spike
        if breakout_high and price_above_hma and vol_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Break below Donchian Low + price below weekly HMA + volume spike
        elif breakout_low and price_below_hma and vol_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals