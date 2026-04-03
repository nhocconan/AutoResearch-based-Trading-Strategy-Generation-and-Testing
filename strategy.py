#!/usr/bin/env python3
"""
Experiment #320: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d HMA trend direction and 
4h volume spike, capture strong momentum moves in both bull and bear markets. The 1d HMA 
provides smooth higher timeframe trend direction, reducing false breakouts during counter-trend 
moves. Volume confirmation ensures institutional participation. Targets 19-50 trades/year on 4h 
timeframe (75-200 total over 4 years) to minimize fee drag while capturing high-probability 
trend continuations. Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel (20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio = volume / vol_ma
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
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(hma_21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 1d HMA direction ---
        # For breakout signals, we need prior HMA value
        if i == 0:
            hma_trend_up = True  # Default for first bar
            hma_trend_down = False
        else:
            hma_trend_up = hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1]
            hma_trend_down = hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss or time-based exit) ---
        if in_position:
            # Stoploss: 2.5 * ATR against position
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if trend reverses strongly (3 consecutive HMA turns)
                if i >= entry_bar + 3 and hma_trend_down:
                    # Check if HMA has been down for 3 bars
                    if (hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] and 
                        hma_21_1d_aligned[i-1] < hma_21_1d_aligned[i-2] and
                        hma_21_1d_aligned[i-2] < hma_21_1d_aligned[i-3]):
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
                # Exit if trend reverses strongly (3 consecutive HMA turns)
                if i >= entry_bar + 3 and hma_trend_up:
                    # Check if HMA has been up for 3 bars
                    if (hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] and 
                        hma_21_1d_aligned[i-1] > hma_21_1d_aligned[i-2] and
                        hma_21_1d_aligned[i-2] > hma_21_1d_aligned[i-3]):
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
            
            # Time-based exit: max 10 bars (~40 hours on 4h)
            if i - entry_bar >= 10:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian HIGH with volume + HMA uptrend
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above channel
            volume_spike and                 # Volume confirmation
            hma_trend_up                     # Higher timeframe uptrend
        )
        
        # Short: Price breaks below Donchian LOW with volume + HMA downtrend
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below channel
            volume_spike and                 # Volume confirmation
            hma_trend_down                   # Higher timeframe downtrend
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>