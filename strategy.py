#!/usr/bin/env python3
"""
Experiment #643: 4h Donchian(20) breakout + 12h/1d HTF filters + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 12h Camarilla pivot levels and 1d EMA200 trend capture institutional flow with minimal trades. Volume confirmation ensures participation. Designed for 4h timeframe to keep trades 75-200 over 4 years, working in bull/bear via HTF regime filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_643_4h_donchian20_12h_camarilla_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3_12h = close_12h + (range_12h * 1.1 / 2.0)
    s3_12h = close_12h - (range_12h * 1.1 / 2.0)
    r4_12h = close_12h + (range_12h * 1.1)
    s4_12h = close_12h - (range_12h * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === HTF: 1d data for EMA200 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # sufficient for 1d EMA200
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_12h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- HTF Regime Filters ---
        # 12h Camarilla: Fade at R3/S3, breakout at R4/S4
        # 1d EMA200: price above = bull regime, below = bear regime
        price_vs_ema = price > ema_1d_aligned[i]  # True = bull regime
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~24h on 4h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + price above 1d EMA200 (bull regime) +
            #       either Camarilla breakout continuation (R4) or fade at S3
            if breakout_up and price_vs_ema:
                if (breakout_up and price > r4_12h_aligned[i]) or \
                   (breakout_up and price < s3_12h_aligned[i] * 1.02):
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            # Short: Donchian breakout down + price below 1d EMA200 (bear regime) +
            #        either Camarilla breakout continuation (S4) or fade at R3
            elif breakout_down and not price_vs_ema:
                if (breakout_down and price < s4_12h_aligned[i]) or \
                   (breakout_down and price > r3_12h_aligned[i] * 0.98):
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals