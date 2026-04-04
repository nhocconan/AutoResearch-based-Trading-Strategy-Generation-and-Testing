#!/usr/bin/env python3
"""
Experiment #4367: 6h Camarilla Pivot + Volume Spike + Regime Filter
HYPOTHESIS: Camarilla pivot levels from 1d provide precise intraday support/resistance. Fade at R3/S3 levels with volume confirmation (>2x average) captures mean reversion in ranging markets, while breakouts at R4/S4 with volume capture continuation in trending markets. Regime filter uses 6h ADX(14) < 20 for ranging (mean revert) and ADX > 25 for trending (breakout). This adaptive approach works in both bull (buying dips at R3/S3) and bear (selling rallies at R3/S3) markets by fading extremes, and captures strong moves via breakouts. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4367_6h_camarilla_pivot_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d OHLC for Camarilla pivot points ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate Camarilla pivot levels from previous day
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_ = prev_high - prev_low
        r3 = pivot + (range_ * 1.1 / 2)
        s3 = pivot - (range_ * 1.1 / 2)
        r4 = pivot + (range_ * 1.1)
        s4 = pivot - (range_ * 1.1)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    
    warmup = max(20, 20, 14)  # vol MA, ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if stoploss hit
            if position_side > 0 and price <= stop_loss:  # Long stop
                in_position = False
                position_side = 0
                signals[i] = 0.0
            elif position_side < 0 and price >= stop_loss:  # Short stop
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE * position_side
            continue
        
        # --- New Position Entry Logic ---
        volume_confirm = vol_ratio[i] > 2.0  # Volume spike > 2x average
        
        # Regime: ADX < 20 = ranging (mean revert), ADX > 25 = trending (breakout)
        ranging = adx[i] < 20
        trending = adx[i] > 25
        
        # Ranging market: fade at R3/S3
        if ranging and volume_confirm:
            # Long near S3 support
            if price <= s3_aligned[i] * 1.002:  # Within 0.2% of S3
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = entry_price - 1.5 * (r3_aligned[i] - s3_aligned[i])  # 1.5x range
                signals[i] = SIZE
            # Short near R3 resistance
            elif price >= r3_aligned[i] * 0.998:  # Within 0.2% of R3
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = entry_price + 1.5 * (r3_aligned[i] - s3_aligned[i])  # 1.5x range
                signals[i] = -SIZE
        
        # Trending market: breakout at R4/S4
        elif trending and volume_confirm:
            # Long breakout above R4
            if price > r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = entry_price - 2.0 * (r4_aligned[i] - r3_aligned[i])  # 2x range
                signals[i] = SIZE
            # Short breakdown below S4
            elif price < s4_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = entry_price + 2.0 * (s3_aligned[i] - s4_aligned[i])  # 2x range
                signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals