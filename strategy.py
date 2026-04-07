#!/usr/bin/env python3
"""
12h Camarilla Pivot + 1d Volume Spike + 1w ADX Regime Filter
Long when price touches Camarilla S3/S4 with volume spike in bullish regime (ADX < 25)
Short when price touches Camarilla R3/R4 with volume spike in bearish regime (ADX > 25)
Exit when price moves back to pivot point (PP) or opposite Camarilla level
Designed for low-frequency, high-conviction trades in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_1w_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla Pivot Levels (from previous day) ===
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    # Avoid division by zero
    range_ = np.where(range_ == 0, 1e-10, range_)
    
    # Camarilla levels
    S1 = prev_close - range_ * 1.0833 / 2
    S2 = prev_close - range_ * 1.1666 / 2
    S3 = prev_close - range_ * 1.2500 / 2
    S4 = prev_close - range_ * 1.5000 / 2
    R1 = prev_close + range_ * 1.0833 / 2
    R2 = prev_close + range_ * 1.1666 / 2
    R3 = prev_close + range_ * 1.2500 / 2
    R4 = prev_close + range_ * 1.5000 / 2
    PP = (prev_high + prev_low + prev_close) / 3
    
    # Align to 12h timeframe
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    PP_12h = align_htf_to_ltf(prices, df_1d, PP)
    
    # === 1d Volume Spike Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    # === 1w ADX Regime Filter ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate ADX (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    # First values
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 12h
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Regime: ADX < 25 = ranging (good for mean reversion), ADX > 25 = trending
    # We'll use ranging regime for mean reversion at Camarilla extremes
    ranging_regime = adx_12h < 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        if (np.isnan(S3_12h[i]) or np.isnan(R3_12h[i]) or np.isnan(PP_12h[i]) or
            np.isnan(vol_spike[i]) or np.isnan(ranging_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to PP or reaches R4 (take profit)
            if close[i] >= PP_12h[i] or close[i] >= R4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to PP or reaches S4 (take profit)
            if close[i] <= PP_12h[i] or close[i] <= S4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in ranging regime (ADX < 25)
            if ranging_regime[i]:
                # Long when price touches S3 or S4 with volume spike
                if vol_spike[i] and (close[i] <= S3_12h[i] or close[i] <= S4_12h[i]):
                    position = 1
                    signals[i] = 0.25
                # Short when price touches R3 or R4 with volume spike
                elif vol_spike[i] and (close[i] >= R3_12h[i] or close[i] >= R4_12h[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals