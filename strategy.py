#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Refined
Hypothesis: Refined version of Camarilla pivot breakout with stricter volume confirmation (3x average volume) and ATR-based position sizing to reduce trade frequency and improve risk-adjusted returns. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Targets 25-40 trades/year to stay within optimal range for 4h timeframe.
"""

name = "4h_1d_Camarilla_Pivot_Refined"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- ATR for position sizing and stop ---
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_close = np.abs(low_4h - np.roll(close_4h, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # --- Camarilla Pivots from 1d (previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots from previous day's data
    camarilla_high = np.full_like(close_1d, np.nan)
    camarilla_low = np.full_like(close_1d, np.nan)
    camarilla_close = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's pivots
        camarilla_high[i] = high_1d[i-1]
        camarilla_low[i] = low_1d[i-1]
        camarilla_close[i] = close_1d[i-1]
    
    # Calculate Camarilla levels
    R4 = camarilla_close + ((camarilla_high - camarilla_low) * 1.5000)
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    R2 = camarilla_close + ((camarilla_high - camarilla_low) * 1.1666)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    PP = camarilla_close
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    S2 = camarilla_close - ((camarilla_high - camarilla_low) * 1.1666)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    S4 = camarilla_close - ((camarilla_high - camarilla_low) * 1.5000)
    
    # Align pivots to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    
    # --- Volume Confirmation: 4h volume > 3x 20-period average (stricter) ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation: at least 3x average volume (much stricter)
        vol_ok = volume_4h[i] > (vol_ma_20[i] * 3.0)
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_4h[i] > R3_4h[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume spike
                # Size based on ATR volatility (inverse volatility scaling)
                vol_factor = min(2.0, max(0.5, 1.0 / (atr[i] / close_4h[i] * 100)))
                base_size = 0.25
                signal_size = base_size * vol_factor
                # Clamp to reasonable range
                signal_size = max(0.15, min(0.35, signal_size))
                signals[i] = signal_size
                position = 1
            elif close_4h[i] < S3_4h[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume spike
                vol_factor = min(2.0, max(0.5, 1.0 / (atr[i] / close_4h[i] * 100)))
                base_size = 0.25
                signal_size = base_size * vol_factor
                signal_size = max(0.15, min(0.35, signal_size))
                signals[i] = -signal_size
                position = -1
        else:
            # Exit conditions with volatility-adjusted bands
            if position == 1:
                # Exit long: price returns to S1 (opposite side) or ATR-based stop
                atr_stop = 1.5 * atr[i]
                if close_4h[i] <= S1_4h[i] or close_4h[i] <= (np.maximum.accumulate(high_4h[:i+1])[-1] - atr_stop):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = signal_size if 'signal_size' in locals() else 0.25
            elif position == -1:
                # Exit short: price returns to R1 (opposite side) or ATR-based stop
                atr_stop = 1.5 * atr[i]
                if close_4h[i] >= R1_4h[i] or close_4h[i] >= (np.minimum.accumulate(low_4h[:i+1])[-1] + atr_stop):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -signal_size if 'signal_size' in locals() else -0.25
    
    return signals