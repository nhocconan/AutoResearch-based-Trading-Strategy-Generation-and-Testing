#!/usr/bin/env python3
# 12h Camarilla Pivot + Volume Spike + 1d Choppiness Regime
# Hypothesis: Camarilla pivot levels (S1/S2 for longs, R1/R2 for shorts) act as support/resistance.
# Combined with volume spike for momentum confirmation and 1d choppiness regime filter to avoid
# whipsaws in trending markets. Designed for low trade frequency and works in both bull/bear
# by switching between mean-reversion (range) and breakout (trend) based on 1d chop regime.
# Target: 50-150 trades over 4 years on 12h timeframe.

name = "12h_Camarilla_Volume_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import ceil
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index: 0 = trending, 100 = ranging"""
    atr_sum = 0.0
    for i in range(len(high)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) if i > 0 else high[i] - low[i]
        atr_sum += tr
        if i < period:
            atr_sum = 0
    if atr_sum == 0:
        return np.full_like(high, 50.0)
    
    # True range sum over period
    tr_sum = np.zeros_like(high, dtype=float)
    for i in range(len(high)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) if i > 0 else high[i] - low[i]
        if i >= period:
            tr_sum[i] = tr_sum[i-1] + tr - (max(high[i-period] - low[i-period], abs(high[i-period] - close[i-period-1]), abs(low[i-period] - close[i-period-1])) if i-period > 0 else high[i-period] - low[i-period])
        else:
            tr_sum[i] = sum([max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) if j > 0 else high[j] - low[j] for j in range(i+1)])
    
    # Avoid division by zero
    atr_period = tr_sum / period
    atr_period[atr_period == 0] = 1e-10
    
    # Highest high and lowest low over period
    highest_high = np.full_like(high, -np.inf)
    lowest_low = np.full_like(high, np.inf)
    for i in range(len(high)):
        if i >= period:
            highest_high[i] = max(high[i-period:i+1])
            lowest_low[i] = min(low[i-period:i+1])
        else:
            highest_high[i] = max(high[:i+1])
            lowest_low[i] = min(low[:i+1])
    
    # Chop = 100 * log10(sum(tr) / (period * (HH - LL))) / log10(period)
    diff = highest_high - lowest_low
    diff[diff == 0] = 1e-10
    chop = 100 * np.log10(tr_sum / (period * diff)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Data for Pivots and Chop Regime ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Use previous day's close to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    R2 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    S2 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 12h timeframe (wait for daily close)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Choppiness index regime filter (needs 2 extra days for confirmation)
    chop = calculate_choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=2)
    
    # Volume spike (20-period on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_12h[i]) or np.isnan(R2_12h[i]) or np.isnan(S1_12h[i]) or np.isnan(S2_12h[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        # Regime: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (breakout)
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # LONG CONDITIONS
            long_signal = False
            if is_ranging:
                # In ranging market: mean revert at S1/S2 support
                if close[i] <= S1_12h[i] and vol_spike[i]:
                    long_signal = True
            else:  # trending or neutral
                # In trending market: breakout above R1 with volume
                if close[i] > R1_12h[i] and vol_spike[i]:
                    long_signal = True
            
            # SHORT CONDITIONS
            short_signal = False
            if is_ranging:
                # In ranging market: mean revert at R1/R2 resistance
                if close[i] >= R1_12h[i] and vol_spike[i]:
                    short_signal = True
            else:  # trending or neutral
                # In trending market: breakdown below S1 with volume
                if close[i] < S1_12h[i] and vol_spike[i]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # EXIT LONG: opposite signal or reversal
            exit_signal = False
            if is_ranging:
                # Exit when price reaches opposite resistance (R1) or loses momentum
                if close[i] >= R1_12h[i]:
                    exit_signal = True
            else:
                # Exit when price breaks below S1 (failed breakout) or loses volume
                if close[i] < S1_12h[i] or not vol_spike[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # EXIT SHORT: opposite signal or reversal
            exit_signal = False
            if is_ranging:
                # Exit when price reaches opposite support (S1) or loses momentum
                if close[i] <= S1_12h[i]:
                    exit_signal = True
            else:
                # Exit when price breaks above R1 (failed breakdown) or loses volume
                if close[i] > R1_12h[i] or not vol_spike[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals