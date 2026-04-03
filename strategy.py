#!/usr/bin/env python3
"""
Experiment #261: 4h Donchian(20) Breakout + 1d EMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Price breaking Donchian(20) channels on 4h timeframe captures institutional breakouts. 
Filtering with 1d EMA(50) ensures alignment with daily trend direction. Volume spike confirmation 
(volume > 1.5x 20-period average) increases probability of sustainable moves. ATR-based stoploss 
manages risk. Targets 20-50 trades/year on 4h (80-200 total over 4 years) to minimize fee drag 
while capturing high-probability breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_ema_volume_v1"
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
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume Spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        donchian_upper = highest_high_20[i]
        donchian_lower = lowest_low_20[i]
        price = close[i]
        
        # --- Breakout Conditions ---
        bullish_breakout = price > donchian_upper and volume_spike[i]
        bearish_breakout = price < donchian_lower and volume_spike[i]
        
        # --- Trend Filter (1d EMA) ---
        price_above_1d_ema = price > ema_50_1d_aligned[i]
        price_below_1d_ema = price < ema_50_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bullish breakout with price above 1d EMA (bullish alignment)
        if bullish_breakout and price_above_1d_ema:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Bearish breakout with price below 1d EMA (bearish alignment)
        elif bearish_breakout and price_below_1d_ema:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals