#!/usr/bin/env python3
"""
Experiment #027: 6h Weekly Pivot + Volume Spike + Regime Filter

HYPOTHESIS: Weekly pivot levels (from 1w HTF) act as strong support/resistance on 6h timeframe.
Breakouts above R1 or below S1 with volume confirmation (2x average volume) and 
regime filter (ADX > 25 for trending, ADX < 20 for ranging) capture institutional flow.
Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper min_periods."""
    return pd.Series(values).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points (R1, S1, R2, S2, R3, S3, R4, S4)."""
    # Typical price
    tp = (high + low + close) / 3.0
    
    # Pivot point
    pp = tp
    
    # Support and resistance levels
    s1 = 2 * pp - high
    r1 = 2 * pp - low
    s2 = pp - (high - low)
    r2 = pp + (high - low)
    s3 = low - 2 * (high - pp)
    r3 = high + 2 * (pp - low)
    s4 = s3 - (r1 - s1)
    r4 = r3 + (r1 - s1)
    
    return pp, r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w for weekly pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points from HTF data
    wh = df_1w['high'].values.astype(np.float64)
    wl = df_1w['low'].values.astype(np.float64)
    wc = df_1w['close'].values.astype(np.float64)
    
    wp, wr1, ws1, wr2, ws2, wr3, ws3, wr4, ws4 = calculate_weekly_pivot(wh, wl, wc)
    
    # Align weekly pivot points to LTF (6h)
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    wr1_aligned = align_htf_to_ltf(prices, df_1w, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1w, ws1)
    wr2_aligned = align_htf_to_ltf(prices, df_1w, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_1w, ws2)
    wr3_aligned = align_htf_to_ltf(prices, df_1w, wr3)
    ws3_aligned = align_htf_to_ltf(prices, df_1w, ws3)
    wr4_aligned = align_htf_to_ltf(prices, df_1w, wr4)
    ws4_aligned = align_htf_to_ltf(prices, df_1w, ws4)
    
    # === 6h Indicators ===
    adx_14 = calculate_adx(high, low, close, period=14)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(adx_14[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(wp_aligned[i]) or np.isnan(wr1_aligned[i]) or np.isnan(ws1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter ---
        # ADX > 25 = trending market (favor breakouts)
        # ADX < 20 = ranging market (favor mean reversion at pivot)
        is_trending = adx_14[i] > 25
        is_ranging = adx_14[i] < 20
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            stop_hit = False
            
            # Exit conditions based on regime
            if position_side > 0:  # Long position
                # In trending: exit when price touches weekly S1 or R2 (profit target/stop)
                # In ranging: exit when price touches weekly R1 (take profit)
                if is_trending:
                    if low[i] <= ws1_aligned[i] or high[i] >= wr2_aligned[i]:
                        stop_hit = True
                else:  # ranging
                    if high[i] >= wr1_aligned[i]:
                        stop_hit = True
            else:  # Short position
                # In trending: exit when price touches weekly R1 or S2
                # In ranging: exit when price touches weekly S1
                if is_trending:
                    if high[i] >= wr1_aligned[i] or low[i] <= ws2_aligned[i]:
                        stop_hit = True
                else:  # ranging
                    if low[i] <= ws1_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions:
        # Breakout above weekly R1 with volume confirmation
        if close[i] > wr1_aligned[i] and vol_ok:
            # In trending market, favor breakouts
            # In ranging market, only take long if price is below pivot (mean reversion long)
            if is_trending or (is_ranging and close[i] < wp_aligned[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
        # Short conditions:
        # Breakdown below weekly S1 with volume confirmation
        elif close[i] < ws1_aligned[i] and vol_ok:
            # In trending market, favor breakouts
            # In ranging market, only take short if price is above pivot (mean reversion short)
            if is_trending or (is_ranging and close[i] > wp_aligned[i]):
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals