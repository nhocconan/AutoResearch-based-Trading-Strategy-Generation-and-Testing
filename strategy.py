#!/usr/bin/env python3
"""
Experiment #5855: 6h Camarilla pivot reversal + 1d volume spike + chop regime filter
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) act as institutional support/resistance on 6h charts. 
In ranging markets (CHOP > 61.8), price reverses at R3/S3 with volume confirmation. 
In trending markets (CHOP < 38.2), breakouts at R4/S4 continue. Uses 1d Camarilla for structure 
and 6h for execution. Works in bull markets (buy R3/S3 bounces, sell R4/S4 breakdowns) and bear 
markets (sell R3/S3 rallies, buy R4/S4 breakdowns). Volume spike confirms institutional interest. 
Chop regime filter ensures correct context. Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5855_6h_camarilla1d_vol_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla pivots from prior day's OHLC
        # Camarilla formulas: based on previous day's range
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        # Range
        rang = prev_high - prev_low
        
        # Camarilla levels
        r4 = pivot + (rang * 1.1 / 2)
        r3 = pivot + (rang * 1.1 / 4)
        s3 = pivot - (rang * 1.1 / 4)
        s4 = pivot - (rang * 1.1 / 2)
        
        # Align to 6h timeframe
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: Choppy market regime (Chopiness Index) ===
    # Chop = 100 * log10(sum(ATR(14)) / (n * (HHV - LLV))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high).rolling(window=14, min_periods=14).max().values - \
           pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.where(hhvl > 0, hhvl, 1)) / np.log10(14)
    chop = np.where(hhvl > 0, chop, 50.0)  # default to neutral when no range
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 14, 2)  # volume avg, chop, shift(1) for pivots
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(volume_ratio[i]) or np.isnan(chop[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: reverse at opposite level or stoploss ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit: price reaches R4 (take profit) or breaks below S3 (stop and reverse)
                if price >= r4_aligned[i] or price <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit: price reaches S4 (take profit) or breaks above R3 (stop and reverse)
                if price <= s4_aligned[i] or price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime-based entry: chop > 61.8 = range (mean revert at R3/S3), chop < 38.2 = trend (breakout at R4/S4)
        volume_confirmed = volume_ratio[i] > 1.8
        
        if chop[i] > 61.8:  # Ranging market: mean revert at R3/S3
            long_setup = price <= s3_aligned[i] and volume_confirmed
            short_setup = price >= r3_aligned[i] and volume_confirmed
        elif chop[i] < 38.2:  # Trending market: breakout at R4/S4
            long_setup = price >= r4_aligned[i] and volume_confirmed
            short_setup = price <= s4_aligned[i] and volume_confirmed
        else:  # Neutral chop: no trade
            long_setup = False
            short_setup = False
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals