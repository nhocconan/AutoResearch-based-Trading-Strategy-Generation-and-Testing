#!/usr/bin/env python3
"""
Experiment #051: 6h Camarilla Pivot + 1d Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe act as 
intraday support/resistance. In ranging markets (CHOP > 61.8), fade touches 
of R3/S3 for mean reversion. In trending markets (CHOP < 38.2), breakouts 
of R4/S4 continue with volume confirmation. Uses 6h primary timeframe to 
balance trade frequency and signal quality. Discrete sizing (0.25) minimizes 
fee churn. Designed to work in both bull (trend continuation) and bear 
(mean reversion in ranges) markets via regime adaptation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_1d_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: measures whether market is ranging (high) or trending (low)."""
    n = len(close)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)
    
    atr_sum = np.zeros(n)
    for i in range(period-1, n):
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        atr_sum[i] = tr_sum
    
    hh = pd.Series(high).rolling(window=period, min_periods=period).max()
    ll = pd.Series(low).rolling(window=period, min_periods=period).min()
    range_hl = hh - ll
    
    chop = np.where(
        (atr_sum > 0) & (range_hl > 0),
        100 * np.log10(atr_sum / (range_hl * period)) / np.log10(period),
        50.0
    )
    return chop.values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    range_hl = df_1d['high'] - df_1d['low']
    r3 = pivot + range_hl * 1.1 / 2.0
    s3 = pivot - range_hl * 1.1 / 2.0
    r4 = pivot + range_hl * 1.1
    s4 = pivot - range_hl * 1.1
    
    # Align HTF arrays to LTF with shift(1) for completed bars only
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Calculate 1d Choppiness Index for regime filter
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 6h Indicators ===
    # Volume spike detection (2.0x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter ---
        is_ranging = chop_1d_aligned[i] > 61.8  # CHOP > 61.8 = ranging (mean revert)
        is_trending = chop_1d_aligned[i] < 38.2  # CHOP < 38.2 = trending (breakout)
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            exit_signal = False
            
            if position_side > 0:  # Long position
                if is_ranging:
                    # In ranging: exit at S3 (mean reversion target)
                    if close[i] <= s3_aligned[i]:
                        exit_signal = True
                else:  # Trending
                    # In trending: exit at R4 (profit target) or reverse signal
                    if close[i] >= r4_aligned[i] or close[i] <= s3_aligned[i]:
                        exit_signal = True
            else:  # Short position
                if is_ranging:
                    # In ranging: exit at R3 (mean reversion target)
                    if close[i] >= r3_aligned[i]:
                        exit_signal = True
                else:  # Trending
                    # In trending: exit at S4 (profit target) or reverse signal
                    if close[i] <= s4_aligned[i] or close[i] >= r3_aligned[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        vol_condition = vol_ok
        
        if is_ranging:
            # Ranging market: mean reversion at R3/S3
            if vol_condition:
                # Short at R3 with rejection (price < R3 but recovering)
                if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                    # Look for rejection wick: high touched R3 but closed below
                    if high[i] >= r3_aligned[i] * 0.999:  # Touched or pierced R3
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        entry_bar = i
                        signals[i] = -SIZE
                # Long at S3 with rejection (price > S3 but struggling)
                elif close[i] > s3_aligned[i] and close[i] < r3_aligned[i]:
                    # Look for rejection wick: low touched S3 but closed above
                    if low[i] <= s3_aligned[i] * 1.001:  # Touched or pierced S3
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        entry_bar = i
                        signals[i] = SIZE
        else:  # Trending market (CHOP < 38.2)
            # Trending market: breakout continuation at R4/S4
            if vol_condition:
                # Long breakout above R4
                if close[i] > r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_bar = i
                    signals[i] = SIZE
                # Short breakdown below S4
                elif close[i] < s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_bar = i
                    signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals