#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation
# Camarilla levels calculated from previous 1d OHLC: R1, R2, R3, R4, S1, S2, S3, S4
# Logic: Fade at R3/S3 (mean reversion), breakout continuation at R4/S4
# Trend filter: 1d EMA50 (long above, short below) to avoid counter-trend trades
# Volume: current volume > 1.5x 20-period average for confirmation
# Designed to work in both bull (breakouts) and bear (fades at extremes) markets
# Target: 15-25 trades/year to avoid fee drag
name = "6h_Camarilla_Pivot_Reversal_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    R1 = pivot + (range_hl * 1.1 / 12)
    R2 = pivot + (range_hl * 1.1 / 6)
    R3 = pivot + (range_hl * 1.1 / 4)
    R4 = pivot + (range_hl * 1.1 / 2)
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)
    S4 = pivot - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for previous day's close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h ATR for stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_6h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Fade at R3/S3 (mean reversion)
            # Short at R3 if below 1d EMA50
            if close[i] >= R3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
            # Long at S3 if above 1d EMA50
            elif close[i] <= S3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Breakout continuation at R4/S4
            # Long breakout at R4 if above 1d EMA50
            elif close[i] > R4_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown at S4 if below 1d EMA50
            elif close[i] < S4_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches S1 (mean reversion target) or R4 (breakout failure) or 2x ATR stop
            if close[i] <= S1_aligned[i] or close[i] >= R4_aligned[i] or close[i] < close[i-1] - 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches R1 (mean reversion target) or S4 (breakdown failure) or 2x ATR stop
            if close[i] >= R1_aligned[i] or close[i] <= S4_aligned[i] or close[i] > close[i-1] + 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals