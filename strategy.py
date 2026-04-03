#!/usr/bin/env python3
"""
Experiment #286: 4h Donchian(20) Breakout + 1d EMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture medium-term momentum with clear structure. 
Filtering by 1d EMA(50) ensures trades align with higher timeframe trend. Volume confirmation 
increases probability of institutional participation. ATR-based stoploss manages risk. 
Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while capturing 
high-probability breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20-1, n):
            window_high = np.max(high[i-20+1:i+1])
            window_low = np.min(low[i-20+1:i+1])
            donchian_high[i] = window_high
            donchian_low[i] = window_low
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # === Exit Logic (ATR-based stoploss) ===
        if in_position:
            # Calculate ATR(14) for dynamic stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
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
                stop_level = entry_price + 2.0 * atr_14
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
        
        # === Volume Filter: Above average volume ===
        vol_ma_20 = np.zeros(n)
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-20+1:i+1])
        if i < 20 or volume[i] < vol_ma_20[i]:
            signals[i] = 0.0
            continue
        
        # === New Position Entry Logic (Only if Flat) ===
        # Long: Price breaks above Donchian high AND price above 1d EMA (bullish alignment)
        if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = 0.0  # Will calculate ATR on next bar for stoploss
            signals[i] = SIZE
        # Short: Price breaks below Donchian low AND price below 1d EMA (bearish alignment)
        elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = 0.0  # Will calculate ATR on next bar for stoploss
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals