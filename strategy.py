#!/usr/bin/env python3
"""
Experiment #290: 1d Donchian Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Daily Donchian(20) breakouts capture medium-term trends, filtered by weekly HMA(21) alignment and volume spikes (>2x 20-period average) to confirm institutional participation. ATR(14) stoploss manages risk. This structure works in both bull and bear markets by only trading with the higher timeframe trend. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_1w_hma_volume_v1"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).rolling(window=half_len, min_periods=half_len).mean().values
        wma_full = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1w = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume Spike: >2x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(donch_high_20[i]) or 
            np.isnan(donch_low_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout
                if close[i] < donch_low_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian breakout
                if close[i] > donch_high_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian(20) high + price > weekly HMA + volume spike
        long_breakout = close[i] > donch_high_20[i]
        long_trend = close[i] > hma_21_1w_aligned[i]
        long_volume = volume_spike[i]
        
        # Short: Price breaks below Donchian(20) low + price < weekly HMA + volume spike
        short_breakout = close[i] < donch_low_20[i]
        short_trend = close[i] < hma_21_1w_aligned[i]
        short_volume = volume_spike[i]
        
        if long_breakout and long_trend and long_volume:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        elif short_breakout and short_trend and short_volume:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals