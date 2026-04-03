#!/usr/bin/env python3
"""
Experiment #1727: 6h Camarilla Pivot + Volume Spike + ADX Trend Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakouts) combined with volume confirmation (>1.8x average) and ADX trend strength (>25) capture both mean reversion in ranges and breakouts in trends. Works in bull/bear by adapting to regime: fades at extremes in ranging markets (ADX<25), breaks with trend in trending markets (ADX>25). Target: 75-150 total trades over 4 years (19-37/year) by using tight pivot-level entries with volume and trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1727_6h_camarilla_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align HTF pivot levels to 6h timeframe (completed bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: ADX(14) for trend strength ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for ADX and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (using 6h ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate 6h ATR for stoploss
            if i >= 1:
                tr_i = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            else:
                tr_i = high[0] - low[0]
            
            # Calculate ATR using Wilder's smoothing (equivalent to EWM with alpha=1/14)
            if not hasattr(generate_signals, 'atr_prev'):
                generate_signals.atr_prev = tr_i
            else:
                generate_signals.atr_prev = (13 * generate_signals.atr_prev + tr_i) / 14
            atr_i = generate_signals.atr_prev
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_i
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    if hasattr(generate_signals, 'atr_prev'):
                        delattr(generate_signals, 'atr_prev')
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_i
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    if hasattr(generate_signals, 'atr_prev'):
                        delattr(generate_signals, 'atr_prev')
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx[i] > 25
        
        if volume_spike:
            if is_trending:
                # Trending market: breakout continuation at R4/S4
                if price > r4_1d_aligned[i] and close[i-1] <= r4_1d_aligned[i]:
                    # Breakout above R4 with volume -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < s4_1d_aligned[i] and close[i-1] >= s4_1d_aligned[i]:
                    # Breakdown below S4 with volume -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Ranging market: mean reversion at R3/S3
                if price < r3_1d_aligned[i] and close[i-1] >= r3_1d_aligned[i]:
                    # Rejection at R3 with volume -> short (fade the move)
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif price > s3_1d_aligned[i] and close[i-1] <= s3_1d_aligned[i]:
                    # Rejection at S3 with volume -> long (fade the move)
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    # Clean up
    if hasattr(generate_signals, 'atr_prev'):
        delattr(generate_signals, 'atr_prev')
    
    return signals