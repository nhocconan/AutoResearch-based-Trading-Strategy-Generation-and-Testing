#!/usr/bin/env python3
"""
Experiment #450: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Donchian channel breakouts on 1d timeframe capture strong directional moves, 
filtered by 1w HMA(21) trend to ensure alignment with higher timeframe direction. 
Volume confirmation (>1.5x 20-period average) ensures institutional participation. 
ATR-based stoploss (2.5x ATR(14)) manages risk. Targets 7-25 trades/year on 1d 
(30-100 total over 4 years) to minimize fee drag while capturing high-probability 
breakouts in both bull and bear markets. The 1w trend filter prevents counter-trend 
trading during strong reversals.
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
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1w = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(hma_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_1w_aligned[i] > hma_21_1w_aligned[i-1]
        hma_falling = hma_21_1w_aligned[i] < hma_21_1w_aligned[i-1]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian low break (trailing stop)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian high break (trailing stop)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and HMA rising
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            hma_rising
        )
        
        # Short: Price breaks below Donchian low with volume and HMA falling
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            hma_falling
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