#!/usr/bin/env python3
"""
6h_Pivot_Range_Reversal_With_Volume_Squeeze
Hypothesis: In ranging markets (ADX<25), price tends to revert from daily pivot support/resistance levels (S1/R1, S2/R2). 
Entry: Price touches S1/R1 with RSI<30/RSI>70 and volume contraction (<0.8x average) in the direction of the reversal.
Exit: Price reaches daily pivot point or opposite S1/R1 level.
Works in both bull/bear by using mean-reversion logic in ranging markets (ADX filter) and avoiding strong trends.
"""

name = "6h_Pivot_Range_Reversal_With_Volume_Squeeze"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate daily pivot points: P = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Shift by 1 to use previous day's data
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan

    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0

    # Support and resistance levels
    r1 = 2 * pivot - prev_low_1d
    s1 = 2 * pivot - prev_high_1d
    r2 = pivot + (prev_high_1d - prev_low_1d)
    s2 = pivot - (prev_high_1d - prev_low_1d)

    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)

    # ADX filter for ranging markets (ADX < 25)
    # Calculate ADX using 14-period
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros_like(high)
        atr[period] = np.nansum(tr[1:period+1]) if not np.any(np.isnan(tr[1:period+1])) else np.nan
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm[i] * 100) / atr[i]
                minus_di[i] = (minus_dm[i] * 100) / atr[i]
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = abs(plus_di[i] - minus_di[i]) * 100 / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period]) if not np.any(np.isnan(dx[period:2*period])) else np.nan
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx

    adx_1d = calculate_adx(prev_high_1d, prev_low_1d, prev_close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # RSI (14-period) for overbought/oversold
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.nansum(gain[1:period+1]) if not np.any(np.isnan(gain[1:period+1])) else np.nan
        avg_loss[period] = np.nansum(loss[1:period+1]) if not np.any(np.isnan(loss[1:period+1])) else np.nan
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(close)
        rsi = np.zeros_like(close)
        for i in range(period, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
                
        return rsi

    rsi_1d = calculate_rsi(prev_close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume contraction: <0.8x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_contraction = volume < (0.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_contraction[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S1 with RSI oversold, volume contraction, and ranging market (ADX<25)
            if (abs(close[i] - s1_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of S1
                rsi_1d_aligned[i] < 30 and
                volume_contraction[i] and
                adx_1d_aligned[i] < 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R1 with RSI overbought, volume contraction, and ranging market (ADX<25)
            elif (abs(close[i] - r1_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of R1
                  rsi_1d_aligned[i] > 70 and
                  volume_contraction[i] and
                  adx_1d_aligned[i] < 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot point or S2 level
            if (close[i] >= pivot_aligned[i] or close[i] <= s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot point or R2 level
            if (close[i] <= pivot_aligned[i] or close[i] >= r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals