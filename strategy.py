#!/usr/bin/env python3
"""
Experiment #5071: 6h Camarilla Pivot Reversion + Volume Spike + ATR Stoploss
HYPOTHESIS: On 6h timeframe, price tends to revert to the mean when reaching extreme Camarilla levels (R3/S3, R4/S4) derived from prior 1d pivot, especially when confirmed by volume spikes (>2x average). In ranging markets (2022-2024), this mean reversion captures profits at extremes. In trending markets, breakouts beyond R4/S4 with volume confirmation continue the trend. The strategy uses discrete position sizing (0.25) to minimize fee churn and ATR-based trailing stops (2.5x) to manage risk. Designed for 12-37 trades/year on 6h timeframe to overcome fee drag while maintaining statistical significance across BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5071_6h_camarilla_pivot_reversion_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (based on prior day's OHLC) ===
    # Camarilla uses prior day's H, L, C to calculate support/resistance
    if len(df_1d) >= 1:
        # Prior day's OHLC (shifted by 1 to avoid look-ahead)
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Camarilla calculation: based on prior day's range
        # Pivot = (H + L + C) / 3
        # R4 = C + (H-L) * 1.1/2
        # R3 = C + (H-L) * 1.1/4
        # R2 = C + (H-L) * 1.1/6
        # R1 = C + (H-L) * 1.1/12
        # S1 = C - (H-L) * 1.1/12
        # S2 = C - (H-L) * 1.1/6
        # S3 = C - (H-L) * 1.1/4
        # S4 = C - (H-L) * 1.1/2
        
        # Calculate for prior day (shift by 1)
        if len(high_1d) >= 2:
            # Use prior day's values (index i-1 for current day i)
            high_prev = np.concatenate([[np.nan], high_1d[:-1]])
            low_prev = np.concatenate([[np.nan], low_1d[:-1]])
            close_prev = np.concatenate([[np.nan], close_1d[:-1]])
            
            # Camarilla levels based on prior day
            pivot = (high_prev + low_prev + close_prev) / 3.0
            rng = high_prev - low_prev
            r4 = close_prev + rng * 1.1 / 2.0
            r3 = close_prev + rng * 1.1 / 4.0
            r2 = close_prev + rng * 1.1 / 6.0
            r1 = close_prev + rng * 1.1 / 12.0
            s1 = close_prev - rng * 1.1 / 12.0
            s2 = close_prev - rng * 1.1 / 6.0
            s3 = close_prev - rng * 1.1 / 4.0
            s4 = close_prev - rng * 1.1 / 2.0
            
            # Align to 6h timeframe (prior day's levels are valid for current day)
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
            s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
            r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
            s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
        else:
            pivot_aligned = np.full(n, np.nan)
            r3_aligned = np.full(n, np.nan)
            s3_aligned = np.full(n, np.nan)
            r4_aligned = np.full(n, np.nan)
            s4_aligned = np.full(n, np.nan)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation (2x spike) ===
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
    
    warmup = max(20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Camarilla reversion/breakout logic
        # Long: Revert from S3/S4 OR breakout above R4 with volume
        # Short: Revert from R3/R4 OR breakdown below S4 with volume
        # Fade extremes (reversion to mean) when price hits R3/S3
        # Continue trend when price breaks R4/S4 with volume
        
        long_condition = (
            # Mean reversion long from support
            ((price <= s3_aligned[i]) and vol_confirm) or
            # Breakout long above resistance
            ((price >= r4_aligned[i]) and vol_confirm)
        )
        
        short_condition = (
            # Mean reversion short from resistance
            ((price >= r3_aligned[i]) and vol_confirm) or
            # Breakdown short below support
            ((price <= s4_aligned[i]) and vol_confirm)
        )
        
        # Final entry conditions
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals