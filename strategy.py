#!/usr/bin/env python3
"""
Experiment #310: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Daily Donchian(20) breakouts capture intermediate-term trends, confirmed by 1-week 
Hull Moving Average trend direction and volume spike. This structure provides high-probability 
trend continuation entries with clear stoploss at opposite Donchian band. Targets 15-25 trades/year 
on 1d timeframe (60-100 total over 4 years) to minimize fee drag while capturing significant 
trend moves in both bull and bear markets. Weekly HMA filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_1w_hma_vol_v1"
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
        # WMA function for HMA calculation
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        half_len = len(close_1w) // 2
        sqrt_len = int(np.sqrt(len(close_1w)))
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, len(close_1w))
        wma_sqrt = wma(close_1w, sqrt_len)
        
        # Handle array alignment for HMA
        hma_1w = np.full(len(close_1w), np.nan)
        if half_len > 0 and sqrt_len > 0 and len(wma_half) >= half_len and len(wma_full) >= len(close_1w) and len(wma_sqrt) >= sqrt_len:
            hma_values = 2 * wma_half[-len(close_1w):] - wma_full
            hma_1w[-sqrt_len:] = wma_sqrt[-sqrt_len:] * 2 - hma_values[-sqrt_len:]
        
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, 1.0)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or Donchian reversal) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                tr[0] = high[0] - low[0]
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
                
                if position_side > 0:  # Long position
                    stop_level = entry_price - 2.5 * atr_14
                    if low[i] < stop_level or close[i] <= donchian_l[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                else:  # Short position
                    stop_level = entry_price + 2.5 * atr_14
                    if high[i] > stop_level or close[i] >= donchian_h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                
                # Hold position
                signals[i] = position_side * SIZE
                continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian H with volume and 1w HMA up
        long_condition = (
            close[i] > donchian_h[i] and 
            vol_ratio[i] > 1.5 and 
            hma_1w_aligned[i] > hma_1w_aligned[i-1]  # HMA rising
        )
        
        # Short: Price breaks below Donchian L with volume and 1w HMA down
        short_condition = (
            close[i] < donchian_l[i] and 
            vol_ratio[i] > 1.5 and 
            hma_1w_aligned[i] < hma_1w_aligned[i-1]  # HMA falling
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals