#!/usr/bin/env python3
"""
Experiment #275: 6h Williams %R + Weekly Pivot Structure + Volume Spike

HYPOTHESIS: Williams %R(14) identifies overextension on 6h timeframe, while weekly pivot levels (from 1w data) provide institutional support/resistance structure. Long when %R < -80 (oversold) AND price above weekly pivot (bullish bias), short when %R > -20 (overbought) AND price below weekly pivot (bearish bias). Volume spike (>1.5x 20-period average) confirms institutional participation. This combines mean reversion extrapolation with structural pivot levels to work in both bull (buying oversold dips above pivot) and bear (selling overbought rallies below pivot) markets. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    if len(df_1w) >= 1:
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Weekly pivot point and support/resistance levels
        weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_r1 = 2 * weekly_pp - weekly_low
        weekly_s1 = 2 * weekly_pp - weekly_high
        weekly_r2 = weekly_pp + (weekly_high - weekly_low)
        weekly_s2 = weekly_pp - (weekly_high - weekly_low)
        weekly_r3 = weekly_high + 2 * (weekly_pp - weekly_low)
        weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pp)
        
        # Align to 6h timeframe (use previous week's levels - no look-ahead)
        weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
        weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
        weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
        weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
        weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    else:
        weekly_pp_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
        weekly_r2_aligned = np.full(n, np.nan)
        weekly_s2_aligned = np.full(n, np.nan)
        weekly_r3_aligned = np.full(n, np.nan)
        weekly_s3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    valid = (highest_high_14 != lowest_low_14) & ~(np.isnan(highest_high_14) | np.isnan(lowest_low_14))
    williams_r[valid] = -100 * (highest_high_14[valid] - close[valid]) / (highest_high_14[valid] - lowest_low_14[valid])
    
    # Volume Spike (>1.5x 20-period average)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
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
                # Take profit at weekly R1 or when %R > -20 (mean reversion)
                if high[i] >= weekly_r1_aligned[i] or williams_r[i] > -20:
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
                # Take profit at weekly S1 or when %R < -80 (mean reversion)
                if low[i] <= weekly_s1_aligned[i] or williams_r[i] < -80:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Oversold Williams %R with price above weekly pivot AND volume spike
        if (williams_r[i] < -80 and 
            close[i] > weekly_pp_aligned[i] and 
            volume_spike[i]):
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Overbought Williams %R with price below weekly pivot AND volume spike
        elif (williams_r[i] > -20 and 
              close[i] < weekly_pp_aligned[i] and 
              volume_spike[i]):
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals