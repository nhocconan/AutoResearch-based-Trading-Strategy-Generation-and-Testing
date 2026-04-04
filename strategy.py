#!/usr/bin/env python3
"""
Experiment #5043: 4h Donchian(20) Breakout + 12h/1d Weekly Pivot Direction + Volume Spike + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with weekly pivot levels (from 12h/1d HTF) capture strong momentum with lower frequency. Weekly pivot acts as regime filter: R3/S3 for mean reversion, R4/S4 for breakout confirmation. Volume > 2x average confirms institutional participation. ATR(14) trailing stop (2.5x) manages risk. Designed for 19-50 trades/year on 4h timeframe to minimize fee drag while maintaining statistical significance. Weekly pivot provides structural support/resistance that works in both bull (breakouts through R4) and bear (breakdowns through S4) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5043_4h_donchian20_12h_1d_weekly_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h and 1d data for weekly pivot levels
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's OHLC) ===
    if len(df_1d) >= 5:  # Need at least a week of data
        # Calculate weekly OHLC from daily data
        high_5d = pd.Series(high).rolling(window=5, min_periods=5).max().values
        low_5d = pd.Series(low).rolling(window=5, min_periods=5).min().values
        close_5d = pd.Series(close).rolling(window=5, min_periods=5).last().values
        
        # Weekly Pivot Point = (Prior Week H + L + C) / 3
        pp = (high_5d + low_5d + close_5d) / 3.0
        
        # Weekly Support/Resistance Levels
        rng = high_5d - low_5d
        r3 = high_5d + 2 * (pp - low_5d)  # R3
        s3 = low_5d - 2 * (high_5d - pp)  # S3
        r4 = pp + 3 * rng                 # R4
        s4 = pp - 3 * rng                 # S4
        
        # Align to 4h timeframe from 1d
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (2x spike) ===
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
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with weekly pivot alignment
        # Long: Donchian breakout above R4 (strong breakout) OR above R3 with volume (mean reversion fail)
        # Short: Donchian breakdown below S4 (strong breakdown) OR below S3 with volume (mean reversion fail)
        breakout_long = ((price >= high_roll[i]) and 
                        ((price >= r4_aligned[i]) or  # Strong breakout through weekly R4
                         ((price >= r3_aligned[i]) and vol_confirm)) and  # Fade failure at R3 with volume
                        vol_confirm)
        
        breakout_short = ((price <= low_roll[i]) and 
                         ((price <= s4_aligned[i]) or  # Strong breakdown through weekly S4
                          ((price <= s3_aligned[i]) and vol_confirm)) and  # Fade failure at S3 with volume
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