#!/usr/bin/env python3
"""
Experiment #2811: 6h Camarilla Pivot Reversal + 1d Volume Spike + ADX Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from daily timeframe act as strong support/resistance.
In ranging markets, price reverses at R3/S3 with volume confirmation. In trending markets (ADX>25),
breakouts at R4/S4 continue with volume spike. This adaptive approach works in both bull and bear
markets by switching between mean reversion and trend following based on regime. 6h timeframe
balances trade frequency and capture of multi-day moves. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2811_6h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and ADX (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate ADX(14) on 1d for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = (np.nansum(plus_dm[i-period+1:i+1]) / atr[i]) * 100
                minus_di[i] = (np.nansum(minus_dm[i-period+1:i+1]) / atr[i]) * 100
                if (plus_di[i] + minus_di[i]) > 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike detection on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones_like(volume_1d)
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h Indicators: None needed, using price action directly ===
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions based on regime
            if adx_1d_aligned[i] > 25:  # Trending regime - trail with 2*ATR
                # Use 6h ATR approximation from Donchian width
                lookback = min(20, i+1)
                highest_6h = np.max(high[i-lookback+1:i+1])
                lowest_6h = np.min(low[i-lookback+1:i+1])
                donchian_width = highest_6h - lowest_6h
                atr_estimate = donchian_width * 0.15
                
                if position_side > 0:  # Long
                    if price < entry_price - 2.0 * atr_estimate:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    if price > entry_price + 2.0 * atr_estimate:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            else:  # Ranging regime - exit at opposite Camarilla level
                if position_side > 0:  # Long
                    if price < s3_1d_aligned[i]:  # Exit at S3
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    if price > r3_1d_aligned[i]:  # Exit at R3
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        if volume_spike:
            # Determine regime
            if adx_1d_aligned[i] > 25:  # Trending regime - breakout continuation
                # Long breakout at R4
                if price > r4_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                # Short breakdown at S4
                elif price < s4_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
            else:  # Ranging regime - mean reversion at R3/S3
                # Long reversal at S3
                if price < s3_1d_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                # Short reversal at R3
                elif price > r3_1d_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals