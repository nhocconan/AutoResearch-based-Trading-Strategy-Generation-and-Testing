#!/usr/bin/env python3
"""
Experiment #151: 6h Camarilla Pivot + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: Camarilla pivot levels on 1d timeframe provide intraday support/resistance.
Breakouts above R4 or below S4 with volume confirmation (>1.5x 20-period average volume)
indicate strong momentum continuation. Choppiness index (1d) filter avoids false breakouts
in ranging markets (CHOP > 61.8 = range, avoid breakouts). Works in both bull/bear via
volatility expansion breakouts that capture strong directional moves regardless of trend.
Target: 75-150 total trades over 4 years (19-37/year) - within winning range for 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot, volume, and choppiness ===
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot levels from previous day
    # Formula based on OHLC of previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot and levels (using previous day's OHLC)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = close_1d + range_1d * 1.500
    r3_1d = close_1d + range_1d * 1.250
    s3_1d = close_1d - range_1d * 1.250
    s4_1d = close_1d - range_1d * 1.500
    
    # Align to 6h timeframe (using previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume spike filter on 1d
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_vol_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Choppiness index on 1d (to avoid ranging markets)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low)) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    price_range = highest_high_1d - lowest_low_1d
    chop_1d = np.where(
        (atr_sum > 0) & (price_range > 0),
        100 * np.log10(atr_sum) / np.log10(14) / np.log10(price_range),
        50.0  # neutral when undefined
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # --- Camarilla Breakout + Volume + Regime Filter ---
        # Avoid breakouts in ranging markets (CHOP > 61.8)
        in_trending_regime = chop_1d_aligned[i] <= 61.8
        
        # Upper breakout: price breaks above R4 with volume spike and trending regime
        upper_breakout = (close[i] > r4_1d_aligned[i]) and vol_spike_1d_aligned[i] and in_trending_regime
        # Lower breakout: price breaks below S4 with volume spike and trending regime
        lower_breakout = (close[i] < s4_1d_aligned[i]) and vol_spike_1d_aligned[i] and in_trending_regime
        
        # --- Position Management ---
        if in_position:
            # Check stoploss: 2.5 * ATR against position
            if position_side > 0:  # Long
                if close[i] < entry_price - 2.5 * atr_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if close[i] > entry_price + 2.5 * atr_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Still in position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: upper breakout + volume confirmation + trending regime
        if upper_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: lower breakout + volume confirmation + trending regime
        elif lower_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals