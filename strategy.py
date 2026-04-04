#!/usr/bin/env python3
"""
Experiment #3255: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture medium-term trends. Weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) provide institutional reference points. Volume spike (>2.0x 20-period average) confirms institutional participation. Works in bull markets via breakout continuation at R4/S4 and in bear markets via mean reversion at R3/S3. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3255_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for weekly
    def camarilla(high, low, close):
        # Typical price
        typical = (high + low + close) / 3
        # Range
        rng = high - low
        # Camarilla levels
        r4 = close + rng * 1.1 / 2
        r3 = close + rng * 1.1 / 4
        s3 = close - rng * 1.1 / 4
        s4 = close - rng * 1.1 / 2
        return r3, r4, s3, s4
    
    r3_1w, r4_1w, s3_1w, s4_1w = camarilla(high_1w, low_1w, close_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, lookback, 20, 14, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: price re-enters Donchian channel or hits Camarilla extreme
            if position_side > 0:  # Long
                # Exit if price drops back into Donchian channel (mean reversion)
                if price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly R4 (breakout exhaustion)
                elif price >= r4_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price rises back into Donchian channel (mean reversion)
                if price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly S4 (breakdown exhaustion)
                elif price <= s4_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND below weekly R3 (mean reversion zone)
            # OR price breaks above Donchian high AND above weekly R4 (breakout continuation)
            if price > highest_high[i]:
                if price < r3_1w_aligned[i] or price > r4_1w_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND above weekly S3 (mean reversion zone)
            # OR price breaks below Donchian low AND below weekly S4 (breakdown continuation)
            elif price < lowest_low[i]:
                if price > s3_1w_aligned[i] or price < s4_1w_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals