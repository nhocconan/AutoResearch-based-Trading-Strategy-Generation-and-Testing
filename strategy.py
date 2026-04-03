#!/usr/bin/env python3
"""
Experiment #091: 6h Camarilla Pivot + 1d Volume Spike + Choppiness Regime Filter

HYPOTHESIS: On 6h timeframe, Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
provide high-probability entry zones when combined with 1d volume spikes (>2x average) and 
choppiness regime filter (CHOP > 61.8 = range for mean reversion, CHOP < 38.2 = trend for breakout). 
This strategy avoids whipsaw by only trading in clear regimes and uses discrete position sizing 
(0.25) to minimize fee drag. Works in both bull and bear markets by adapting to regime conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_chop_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots, volume MA, and choppiness (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values.astype(np.float64)
    high_1d = df_1d['high'].values.astype(np.float64)
    low_1d = df_1d['low'].values.astype(np.float64)
    volume_1d = df_1d['volume'].values.astype(np.float64)
    
    # Calculate 1d indicators
    # Previous day's OHLC for Camarilla pivots
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_range = prev_high - prev_low
    
    # Camarilla levels (based on previous day)
    camarilla_h4 = prev_close + prev_range * 1.1 / 2
    camarilla_l4 = prev_close - prev_range * 1.1 / 2
    camarilla_h3 = prev_close + prev_range * 1.1 / 4
    camarilla_l3 = prev_close - prev_range * 1.1 / 4
    
    # 1d volume MA (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) on 1d
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Choppiness Index: 100 * log10(sum(TR)/ (ATR * period)) / log10(period)"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.concatenate([[np.nan], close_arr[:-1]]))
        tr3 = np.abs(low_arr - np.concatenate([[np.nan], close_arr[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
        sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum()
        chop = 100 * np.log10(sum_tr / (atr * period)) / np.log10(period)
        return chop.values
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align all 1d indicators to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 6h Indicators ===
    atr_14 = pd.Series(
        np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.concatenate([[np.nan], close[:-1]])),
                np.abs(low - np.concatenate([[np.nan], close[:-1]]))
            )
        )
    ).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Conditions ---
        vol_ok = volume[i] > vol_ma_1d_aligned[i] * 2.0 if vol_ma_1d_aligned[i] > 1e-10 else False  # 2x volume spike
        chop_value = chop_1d_aligned[i]
        is_choppy = chop_value > 61.8  # Range regime
        is_trending = chop_value < 38.2  # Trend regime
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit conditions based on regime
            if position_side > 0:  # Long position
                if is_choppy:
                    # In range: exit at opposite Camarilla level (S3)
                    if low[i] <= camarilla_l3_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Trending
                    # In trend: exit on trend reversal or volume dry-up
                    if high[i] >= camarilla_h4_aligned[i] or not vol_ok:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
            else:  # Short position
                if is_choppy:
                    # In range: exit at opposite Camarilla level (R3)
                    if high[i] >= camarilla_h3_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:  # Trending
                    # In trend: exit on trend reversal or volume dry-up
                    if low[i] <= camarilla_l4_aligned[i] or not vol_ok:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if vol_ok:
            if is_choppy:
                # Range regime: mean reversion at R3/S3
                if high[i] >= camarilla_h3_aligned[i] and low[i] <= camarilla_h3_aligned[i]:
                    # Price touching R3 - short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                elif low[i] <= camarilla_l3_aligned[i] and high[i] >= camarilla_l3_aligned[i]:
                    # Price touching S3 - long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
            elif is_trending:
                # Trend regime: breakout continuation at R4/S4
                if high[i] > camarilla_h4_aligned[i]:
                    # Breakout above R4 - long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                elif low[i] < camarilla_l4_aligned[i]:
                    # Breakdown below S4 - short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals