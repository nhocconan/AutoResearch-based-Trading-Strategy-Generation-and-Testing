#!/usr/bin/env python3
"""
Experiment #1950: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture institutional momentum. 
Weekly HMA(21) filter ensures alignment with primary trend. Volume > 1.5x 20-day average confirms institutional participation. 
ATR-based stoploss manages risk. Works in bull/bear by following weekly trend direction. 
Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1950_1d_donchian20_1w_hma_vol_v1"
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
    close_1w = df_1w['close'].values
    
    # Calculate weekly HMA(21)
    def calculate_hma(arr, period):
        half_len = period // 2
        sqrt_len = int(np.sqrt(period))
        if half_len < 1:
            half_len = 1
        if sqrt_len < 1:
            sqrt_len = 1
        # WMA of half period
        wma_half = pd.Series(arr).ewm(span=half_len, adjust=False).mean().values
        # WMA of full period
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        # Raw HMA = 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        # Final HMA = WMA of raw_hma with sqrt period
        hma = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        return hma
    
    hma_21_1w = calculate_hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_21_1w)
    
    # === 1d Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_upper = high_roll
    dc_lower = low_roll
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 50  # sufficient for Donchian(20), volume MA, ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Stoploss: 2*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > entry_price + 2.0 * entry_atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Exit on opposite Donchian break (mean reversion)
            if position_side > 0 and price < dc_lower[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price > dc_upper[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly trend alignment
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND weekly trend up
            if trend_bias > 0 and price > dc_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND weekly trend down
            elif trend_bias < 0 and price < dc_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals