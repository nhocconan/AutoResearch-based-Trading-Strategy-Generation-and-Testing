#!/usr/bin/env python3
"""
Experiment #179: 6h Camarilla Pivot + Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels on 12h timeframe provide reliable support/resistance zones. 
Breakouts above R4 or below S4 with volume confirmation (>1.5x average) and regime filter 
(ADX > 25 for trending markets) capture strong momentum moves. In ranging markets (ADX < 20),
we fade at R3/S3 levels with volume confirmation. This adaptive approach works in both bull 
and bear markets by switching between breakout and mean-reversion based on volatility regime.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
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
    
    # === HTF: 12h data for Camarilla pivot and ADX regime (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from 12h OHLC
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    if len(df_12h) >= 2:
        close_12h = df_12h['close'].values
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        for i in range(len(df_12h)):
            # Camarilla calculations using previous 12h bar
            if i == 0:
                # For first bar, use same bar (will be overwritten by alignment)
                pp = (high_12h[i] + low_12h[i] + close_12h[i]) / 3
                r = high_12h[i] - low_12h[i]
            else:
                pp = (high_12h[i-1] + low_12h[i-1] + close_12h[i-1]) / 3
                r = high_12h[i-1] - low_12h[i-1]
            
            camarilla_pivot[i] = pp
            camarilla_h4[i] = pp + r * 1.1 / 2
            camarilla_l4[i] = pp - r * 1.1 / 2
            camarilla_h3[i] = pp + r * 1.1 / 4
            camarilla_l3[i] = pp - r * 1.1 / 4
            camarilla_h2[i] = pp + r * 1.1 / 6
            camarilla_l2[i] = pp - r * 1.1 / 6
            camarilla_h1[i] = pp + r * 1.1 / 12
            camarilla_l1[i] = pp - r * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l1)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    
    # Calculate ADX(14) on 12h for regime detection
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr = np.zeros(len(df_12h))
        tr[0] = high_12h[0] - low_12h[0]
        for i in range(1, len(df_12h)):
            tr[i] = max(high_12h[i] - low_12h[i], 
                       abs(high_12h[i] - close_12h[i-1]), 
                       abs(low_12h[i] - close_12h[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros(len(df_12h))
        dm_minus = np.zeros(len(df_12h))
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(df_12h)):
            up_move = high_12h[i] - high_12h[i-1]
            down_move = low_12h[i-1] - low_12h[i]
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0
        
        # Smoothed TR, DM+
        tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
        di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 100 * abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx_12h = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        adx_aligned = np.full(n, 20.0)  # Default to ranging
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], 
                      abs(high[i] - close[i-1]), 
                      abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    
    warmup = 200  # Ensure enough data for HTF indicators and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Detection ---
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # --- Volume Confirmation ---
        volume_spike = vol_ratio[i] > 1.5
        
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
        
        # --- Entry Logic ---
        long_signal = False
        short_signal = False
        
        if is_trending and volume_spike:
            # Trending market: Breakout at R4/S4
            if close[i] > h4_aligned[i]:
                long_signal = True
            elif close[i] < l4_aligned[i]:
                short_signal = True
        elif is_ranging and volume_spike:
            # Ranging market: Fade at R3/S3
            if close[i] > h3_aligned[i]:
                short_signal = True
            elif close[i] < l3_aligned[i]:
                long_signal = True
        
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals