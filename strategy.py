#!/usr/bin/env python3
"""
Experiment #147: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation

HYPOTHESIS: Donchian breakouts capture momentum, while weekly pivot levels (from 1w HTF) 
provide institutional support/resistance. Breakouts in direction of weekly pivot bias 
have higher success rate. Volume confirmation filters false breakouts. 6h timeframe 
reduces noise vs lower TFs. Works in bull/bear via pivot direction filter that adapts 
to longer-term structure. Target: 75-150 total trades over 4 years.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot points ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivots to 6h timeframe (shifted by 1 for completed week only)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = donchian_channel(high, low, 20)
    
    # === 6h Indicators: Volume Spike (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Current volume vs 20-bar average
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Breakout Conditions ---
        bullish_breakout = close[i] > donch_upper[i-1]  # Close above prior upper band
        bearish_breakout = close[i] < donch_lower[i-1]  # Close below prior lower band
        
        # --- Weekly Pivot Direction Filter ---
        # Long bias: price above weekly S3 (bullish territory)
        # Short bias: price below weekly R3 (bearish territory)
        bullish_bias = close[i] > s3_1w_aligned[i]
        bearish_bias = close[i] < r3_1w_aligned[i]
        
        # --- Volume Confirmation ---
        vol_confirm = vol_ratio[i] > 1.5  # Volume at least 1.5x average
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit when opposite breakout occurs or price returns to pivot
            if position_side == 1:  # Long position
                exit_signal = bearish_breakout or close[i] < pivot_1w_aligned[i]
            else:  # Short position
                exit_signal = bullish_breakout or close[i] > pivot_1w_aligned[i]
            
            if exit_signal:
                in_position = False
                position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: bullish breakout + bullish weekly bias + volume confirmation
        if bullish_breakout and bullish_bias and vol_confirm:
            in_position = True
            position_side = 1
            signals[i] = SIZE
        # Short: bearish breakout + bearish bias + volume confirmation
        elif bearish_breakout and bearish_bias and vol_confirm:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
    
    return signals