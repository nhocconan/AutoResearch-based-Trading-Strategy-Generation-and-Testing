#!/usr/bin/env python3
"""
Experiment #007: 6h Camarilla Pivot + Volume Spike + Daily Trend Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) derived from 1d candles
provide high-probability reversal/continuation zones. Combined with volume confirmation (>2x 20-period average)
and daily trend filter (price vs 1d EMA200), the strategy captures both mean-reversion in ranges and 
breakouts in trends. Designed for 6h timeframe to balance trade frequency (~12-37/year) and minimize fee drag
while adapting to both bull and bear regimes via the daily EMA200 filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average with proper min_periods"""
    return pd.Series(close, dtype=np.float64).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and EMA200 (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Camarilla pivot calculation: based on previous day's OHLC
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    pivot = (c_high + c_low + c_close) / 3.0
    range_hl = c_high - c_low
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2.0)
    s3 = pivot - (range_hl * 1.1 / 2.0)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    # Daily EMA200 for trend filter
    ema_200 = calculate_ema(df_1d['close'].values, 200)
    
    # Align HTF values to 6h timeframe (with shift(1) for completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === 6h Indicators ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 200  # Ensure enough data for EMA200 and HTF alignment
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation (>2x 20-period average) ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False
        
        # --- Daily Trend Filter ---
        daily_bullish = close[i] > ema_200_aligned[i]
        daily_bearish = close[i] < ema_200_aligned[i]
        
        # --- Camarilla Logic ---
        # Mean reversion at R3/S3 (fade extreme moves)
        mean_rev_long = (close[i] <= s3_aligned[i]) and daily_bullish
        mean_rev_short = (close[i] >= r3_aligned[i]) and daily_bearish
        # Breakout continuation at R4/S4 (strong momentum)
        breakout_long = (close[i] >= r4_aligned[i]) and daily_bullish
        breakout_short = (close[i] <= s4_aligned[i]) and daily_bearish
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            exit_signal = False
            # Exit conditions: opposite signal or volume drying up
            if position_side > 0:  # Long position
                if mean_rev_short or breakout_short or not vol_ok:
                    exit_signal = True
            else:  # Short position
                if mean_rev_long or breakout_long or not vol_ok:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume confirmation and either mean reversion or breakout in trend direction
        if vol_ok:
            if mean_rev_long or breakout_long:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            elif mean_rev_short or breakout_short:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals