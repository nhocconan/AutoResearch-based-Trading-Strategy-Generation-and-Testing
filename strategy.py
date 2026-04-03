#!/usr/bin/env python3
"""
Experiment #011: 6h Camarilla Pivot + Volume Spike + Regime Filter (ADX)

HYPOTHESIS: Camarilla pivot levels derived from 1d OHLC provide institutional support/resistance
zones. At 6h timeframe, we fade extreme touches of R3/S3 levels (mean reversion in range) and
breakout continuation at R4/S4 levels (trend following). Volume spikes (>2x 20-period MA)
confirm institutional participation. ADX(14) > 25 filters for trending regimes to avoid false
signals in chop. Targets 50-150 trades over 4 years (12-37/year) on 6h timeframe to balance
opportunity capture with fee minimization. Works in bull/bear via regime-adaptive logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and ADX regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC (based on previous day)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    if len(df_1d) >= 2:
        # Use previous day's OHLC to avoid look-ahead
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Shift by 1 to use previous completed day
        high_1d_prev = np.roll(high_1d, 1)
        low_1d_prev = np.roll(low_1d, 1)
        close_1d_prev = np.roll(close_1d, 1)
        high_1d_prev[0] = np.nan
        low_1d_prev[0] = np.nan
        close_1d_prev[0] = np.nan
        
        # Calculate pivot and ranges
        pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
        range_hl = high_1d_prev - low_1d_prev
        
        # Camarilla levels
        camarilla_close = close_1d_prev
        camarilla_h4 = pivot + (range_hl * 1.1 / 2)
        camarilla_l4 = pivot - (range_hl * 1.1 / 2)
        camarilla_h3 = pivot + (range_hl * 1.1 / 4)
        camarilla_l3 = pivot - (range_hl * 1.1 / 4)
        camarilla_h2 = pivot + (range_hl * 1.1 / 6)
        camarilla_l2 = pivot - (range_hl * 1.1 / 6)
        camarilla_h1 = pivot + (range_hl * 1.1 / 12)
        camarilla_l1 = pivot - (range_hl * 1.1 / 12)
        
        # Align to 6h timeframe (shift(1) inside align_htf_to_ltf)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
        camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
        camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
        camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
        camarilla_close_aligned = align_htf_to_ltf(prices, df_1d, camarilla_close)
    else:
        camarilla_h4_aligned = camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_h3_aligned = camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h2_aligned = camarilla_l2_aligned = np.full(n, np.nan)
        camarilla_h1_aligned = camarilla_l1_aligned = np.full(n, np.nan)
        camarilla_close_aligned = np.full(n, np.nan)
    
    # Calculate ADX(14) on 1d for regime filter
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr1[0] = np.nan
        tr2[0] = np.nan
        tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_ma = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_ma = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI values
        di_plus = 100 * dm_plus_ma / tr_ma
        di_minus = 100 * dm_minus_ma / tr_ma
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align ADX to 6h
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        is_trending = adx_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price levels for reference ---
        price = close[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_trending and volume_spike:
            # In trending markets: breakout continuation at H4/L4
            long_breakout = price > h4
            short_breakout = price < l4
            
            if long_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif not is_trending and volume_spike:
            # In ranging markets: mean reversion at H3/L3 (fade extremes)
            long_reversion = price < l3 and price > camarilla_l1_aligned[i]
            short_reversion = price > h3 and price < camarilla_h1_aligned[i]
            
            if long_reversion:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_reversion:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals