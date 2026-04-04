#!/usr/bin/env python3
"""
Experiment #3292: 12h Camarilla Pivot + 1d Volume Spike + Choppiness Regime
HYPOTHESIS: 12h Camarilla pivot levels (L3/H3) act as strong support/resistance. 
Volume spike (>2.0x average) confirms breakout strength. Choppiness index (CHOP > 61.8) 
filters for ranging markets where mean reversion at pivots works best. 
In trending markets (CHOP < 38.2), we fade extreme touches (L4/H4) as exhaustion signals.
Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year).
Designed to work in both bull (trend continuation from L3/H3) and bear (mean reversion at extremes) 
markets by adapting to regime via choppiness filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3292_12h_camarilla_pivot_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L3 = C - (Range * 1.1 / 4)
    # L4 = C - (Range * 1.1 / 2)
    # H3 = C + (Range * 1.1 / 4)
    # H4 = C + (Range * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    L3 = close_1d - (range_1d * 1.1 / 4.0)
    L4 = close_1d - (range_1d * 1.1 / 2.0)
    H3 = close_1d + (range_1d * 1.1 / 4.0)
    H4 = close_1d + (range_1d * 1.1 / 2.0)
    
    # Align HTF pivot levels to 12h timeframe (with shift(1) for completed bars only)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime detection ===
    # CHOP = 100 * log10(sum(ATR(14)) / (log(n) * (HH - LL))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.full(n, np.nan)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Avoid division by zero and log of zero
    hh_ll = highest_high - lowest_low
    valid = (hh_ll > 0) & ~np.isnan(hh_ll) & ~np.isnan(atr_sum) & (atr_sum > 0)
    chop[valid] = 100 * np.log10(atr_sum[valid] / (np.log(14) * hh_ll[valid])) / np.log10(14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 14)  # sufficient for volume MA and CHOP/ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions based on regime
            if chop[i] > 61.8:  # Ranging market - mean reversion
                # Long: exit at L3 (support) or H3 (resistance) - take profit at opposite pivot
                if position_side > 0:  # Long
                    if price >= H3_aligned[i]:  # Hit resistance, take profit
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    elif price <= L3_aligned[i]:  # Stop loss at support
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    if price <= L3_aligned[i]:  # Hit support, take profit
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    elif price >= H3_aligned[i]:  # Stop loss at resistance
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            else:  # Trending market - trend continuation or exhaustion fade
                # In trending markets, we enter on L3/H3 breaks and exit on L4/H4 (extreme exhaustion)
                if position_side > 0:  # Long
                    if price >= H4_aligned[i]:  # Extreme exhaustion - reverse or stop
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    elif price <= L3_aligned[i]:  # Stop loss at support
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    if price <= L4_aligned[i]:  # Extreme exhaustion - reverse or stop
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    elif price >= H3_aligned[i]:  # Stop loss at resistance
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
            if chop[i] > 61.8:  # Ranging market - mean reversion at pivots
                # Long when price touches L3 and bounces (close above L3 after touching or penetrating)
                # Short when price touches H3 and bounces (close below H3 after touching or penetrating)
                if price <= L3_aligned[i] and close[i] > L3_aligned[i]:
                    # Long entry: price found support at L3
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                elif price >= H3_aligned[i] and close[i] < H3_aligned[i]:
                    # Short entry: price found resistance at H3
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Trending market - trend continuation on breakouts
                # Long when price breaks above H3 with volume
                # Short when price breaks below L3 with volume
                if price > H3_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                elif price < L3_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals