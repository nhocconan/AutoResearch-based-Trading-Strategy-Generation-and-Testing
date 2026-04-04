#!/usr/bin/env python3
"""
Experiment #5086: 4h Donchian(20) Breakout + 1d Pivot Direction + Volume Spike + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with daily pivot levels (from 1d HTF) capture strong momentum with lower frequency. Daily pivot acts as regime filter: R1/S1 for mean reversion, R2/S2 for breakout confirmation. Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 19-50 trades/year on 4h timeframe to minimize fee drag while maintaining statistical significance. Daily pivot provides structural support/resistance that works in both bull (breakouts through R2) and bear (breakdowns through S2) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5086_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for daily pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Daily Pivot Points (using prior day's OHLC) ===
    if len(df_1d) >= 1:
        # Use prior day's OHLC for pivot calculation
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Daily Pivot Point = (Prior Day H + L + C) / 3
        pp = (high_1d + low_1d + close_1d) / 3.0
        
        # Daily Support/Resistance Levels
        # R1 = (2 * PP) - Prior Day L
        # S1 = (2 * PP) - Prior Day H
        # R2 = PP + (Prior Day H - Prior Day L)
        # S2 = PP - (Prior Day H - Prior Day L)
        # R3 = Prior Day H + 2*(PP - Prior Day L)
        # S3 = Prior Day L - 2*(Prior Day H - PP)
        rng = high_1d - low_1d
        r1 = (2 * pp) - low_1d
        s1 = (2 * pp) - high_1d
        r2 = pp + rng
        s2 = pp - rng
        r3 = high_1d + 2 * (pp - low_1d)
        s3 = low_1d - 2 * (high_1d - pp)
        
        # Align to 4h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with daily pivot alignment
        # Long: Donchian breakout above R2 (strong breakout) OR above R1 with volume (mean reversion fail)
        # Short: Donchian breakdown below S2 (strong breakdown) OR below S1 with volume (mean reversion fail)
        breakout_long = ((price >= high_roll[i]) and 
                        ((price >= r2_aligned[i]) or  # Strong breakout through daily R2
                         ((price >= r1_aligned[i]) and vol_confirm)) and  # Fade failure at R1 with volume
                        vol_confirm)
        
        breakout_short = ((price <= low_roll[i]) and 
                         ((price <= s2_aligned[i]) or  # Strong breakdown through daily S2
                          ((price <= s1_aligned[i]) and vol_confirm)) and  # Fade failure at S1 with volume
                         vol_confirm)
        
        # Final entry conditions
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