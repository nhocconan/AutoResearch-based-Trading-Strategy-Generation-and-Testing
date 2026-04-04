#!/usr/bin/env python3
"""
Experiment #3299: 6h Camarilla Pivot + 12h Volume Spike + ADX Regime Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe 
provide institutional-grade support/resistance. Volume spike (>2.0x 20-period average) confirms 
institutional participation. ADX (>25) ensures we only trade in trending regimes to avoid 
choppy market whipsaws. Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year).
Designed to work in bull markets (breakout continuation at R4/S4) and bear markets (mean reversion 
from R3/S3 extremes) by adapting logic based on ADX regime and price location relative to pivots.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3299_6h_camarilla_pivot_12h_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h timeframe
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    r4_12h = close_12h + (range_12h * 1.1 / 2.0)
    r3_12h = close_12h + (range_12h * 1.1 / 4.0)
    s3_12h = close_12h - (range_12h * 1.1 / 4.0)
    s4_12h = close_12h - (range_12h * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (with shift(1) for completed bars only)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ADX(14) for regime filtering ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))]
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 14, 14)  # sufficient for volume MA, ADX, and TR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(s4_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion to pivot or opposite Camarilla level ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit if price reaches pivot (mean reversion) or touches R4 (take profit)
                if price <= pivot_12h_aligned[i] or price >= r4_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if price reaches pivot (mean reversion) or touches S4 (take profit)
                if price >= pivot_12h_aligned[i] or price <= s4_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) and trending regime (ADX > 25)
        volume_spike = vol_ratio[i] > 2.0
        trending_regime = adx[i] > 25
        
        if volume_spike and trending_regime:
            # Long entry: price breaks above R3 with bullish momentum (above pivot)
            if price > r3_12h_aligned[i] and price > pivot_12h_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below S3 with bearish momentum (below pivot)
            elif price < s3_12h_aligned[i] and price < pivot_12h_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals