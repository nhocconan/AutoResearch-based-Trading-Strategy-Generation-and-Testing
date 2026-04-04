#!/usr/bin/env python3
"""
Experiment #5263: 4h Donchian(20) Breakout + Volume Spike + Choppiness Regime Filter (12h/1d)
HYPOTHESIS: On 4h timeframe, price breaking Donchian(20) channels with volume confirmation (>1.5x average) and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend) captures high-probability breakouts while avoiding false signals. Uses discrete position sizing (0.25) and ATR-based trailing stop (2.0x) to manage risk. Designed for 25-50 trades/year on 4h timeframe (100-200 total over 4 years) to minimize fee drag. Works in bull markets (breakouts continue uptrend) and bear markets (breakouts continue downtrend) by aligning with higher timeframe structure and regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5263_4h_donchian_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Donchian channels (structure) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        # Donchian(20) on prior completed 12h bar (shift(1) in align)
        donch_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().shift(1).values
        donch_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().shift(1).values
        donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
        donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    else:
        donch_high_aligned = np.full(n, np.nan)
        donch_low_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for choppiness regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # True Range for CHOP calculation
        tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
        tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
        tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
        # We calculate over 14-period window
        sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
        chop_denom = max_high_14 - min_low_14
        chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
        chop_raw = 100 * np.log10(sum_atr_14 / chop_denom_safe) / np.log10(14)
        chop_1d = pd.Series(chop_raw).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 14)  # Donchian, Vol MA, ATR, CHOP warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Regime filter: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid breakouts)
        regime_trending = chop_aligned[i] < 38.2
        
        # Donchian breakout in trending regime only
        breakout_long = (price >= donch_high_aligned[i]) and vol_confirm and regime_trending
        breakout_short = (price <= donch_low_aligned[i]) and vol_confirm and regime_trending
        
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