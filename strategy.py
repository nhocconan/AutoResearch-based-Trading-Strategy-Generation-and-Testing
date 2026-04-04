#!/usr/bin/env python3
"""
Experiment #4911: 6h Camarilla Pivot + 1d Volume Spike + ATR Filter
HYPOTHESIS: On 6h timeframe, price rejection at Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from prior 1d, confirmed by 1d volume spike (>2x average) and filtered by ATR volatility regime, captures institutional reaction to key levels. Works in bull/bear via dual long/short logic. Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4911_6h_camarilla_pivot_1d_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (from prior day) ===
    if len(df_1d) >= 1:
        # Calculate Camarilla levels from previous 1d bar's OHLC
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Camarilla levels
        r3 = pivot + (range_hl * 1.1 / 4.0)
        s3 = pivot - (range_hl * 1.1 / 4.0)
        r4 = pivot + (range_hl * 1.1 / 2.0)
        s4 = pivot - (range_hl * 1.1 / 2.0)
        
        # Align to 6h timeframe (shifted by 1 for completed 1d bar)
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
    
    # === 1d Indicators: Volume Spike (>2x 20-period average) ===
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.ones(len(df_1d))
        vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_1d[20:]
        vol_spike_1d = vol_ratio_1d > 2.0
        vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(np.float64))
    else:
        vol_spike_aligned = np.full(n, 0.0)
    
    # === 6h Indicators: ATR(14) for volatility filter and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === ATR-based volatility regime filter (avoid choppy markets) ===
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.ones(n)
    atr_ratio[20:] = atr[20:] / atr_ma[20:]
    # Only trade when volatility is elevated (ATR > 1.2x average) or normal (not extreme low)
    vol_regime = (atr_ratio >= 0.8) & (atr_ratio <= 3.0)  # Avoid extremely low/high vol
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Camarilla needs 1d, volume MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_spike_aligned[i])):
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
        
        # --- Entry Conditions ---
        # Volume spike confirmation from 1d
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        # Volatility regime filter
        vol_regime_ok = vol_regime[i]
        
        # Mean reversion at R3/S3 (price rejects extreme levels)
        mean_revert_long = (price <= s3_aligned[i]) and (price > s4_aligned[i]) and vol_confirm and vol_regime_ok
        mean_revert_short = (price >= r3_aligned[i]) and (price < r4_aligned[i]) and vol_confirm and vol_regime_ok
        
        # Breakout continuation at R4/S4 (price breaks extreme levels with volume)
        breakout_long = (price >= r4_aligned[i]) and vol_confirm and vol_regime_ok
        breakout_short = (price <= s4_aligned[i]) and vol_confirm and vol_regime_ok
        
        # Final entry logic: mean reversion OR breakout
        if mean_revert_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif mean_revert_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals