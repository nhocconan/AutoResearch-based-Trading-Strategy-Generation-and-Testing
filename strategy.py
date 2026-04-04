#!/usr/bin/env python3
"""
Experiment #4619: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking prior 20-period Donchian channels with volume confirmation (>1.3x avg volume) 
and aligned with weekly pivot direction (price above/below weekly pivot) captures strong momentum breakouts.
Uses weekly HTF for pivot to avoid look-ahead. Discrete sizing (0.25) and ATR trailing stop (2.0x) manage risk.
Target: 12-37 trades/year on 6h timeframe. Works in both bull (breakouts with trend) and bear (breakdowns with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4619_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (standard: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H)
    if len(df_1w) >= 1:
        # Use prior week's OHLC (shifted by 1 to avoid look-ahead)
        wh = np.concatenate([[np.nan], df_1w['high'].values[:-1]])   # prior week high
        wl = np.concatenate([[np.nan], df_1w['low'].values[:-1]])    # prior week low
        wc = np.concatenate([[np.nan], df_1w['close'].values[:-1]])  # prior week close
        
        # Weekly pivot calculations
        weekly_pivot = (wh + wl + wc) / 3.0
        weekly_r1 = 2 * weekly_pivot - wl  # R1 = 2*P - L
        weekly_s1 = 2 * weekly_pivot - wh  # S1 = 2*P - H
    else:
        weekly_pivot = np.array([])
        weekly_r1 = np.array([])
        weekly_s1 = np.array([])
    
    # Align weekly pivot levels to 6h timeframe
    if len(weekly_pivot) > 0:
        pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
        s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    else:
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Upper channel = max(high, 20), Lower channel = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.3x avg volume)
        vol_confirm = vol_ratio[i] > 1.3
        
        # Weekly pivot direction: price above pivot = bullish bias, below = bearish bias
        bullish_bias = price > pivot_aligned[i]
        bearish_bias = price < pivot_aligned[i]
        
        # Breakout conditions: price breaks Donchian channels with volume confirmation and pivot alignment
        breakout_long = price > donchian_upper[i] and vol_confirm and bullish_bias
        breakout_short = price < donchian_lower[i] and vol_confirm and bearish_bias
        
        # Entry logic
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</trading_assistant>