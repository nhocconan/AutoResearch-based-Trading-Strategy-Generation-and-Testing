#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Parabolic SAR trend following with weekly pivot support/resistance
# Uses ADX(14) to identify trending markets (ADX > 25) and Parabolic SAR for entry/exit
# Weekly pivot levels (from 1w) act as major support/resistance - price must be above/below pivot for longs/shorts
# Works in bull markets via buying dips above weekly pivot, in bear via selling rallies below weekly pivot
# Target: 15-35 trades/year to avoid fee drag
name = "6h_ADX_SAR_WeeklyPivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # ADX calculation (14-period)
    # +DM = max(high - previous high, 0) if high - previous high > previous low - low else 0
    # -DM = max(previous low - low, 0) if previous low - low > high - previous high else 0
    # TR = max(high - low, abs(high - previous close), abs(low - previous close))
    # +DM14 = smoothed +DM, -DM14 = smoothed -DM, TR14 = smoothed TR
    # +DI14 = 100 * +DM14 / TR14, -DI14 = 100 * -DM14 / TR14
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed DX
    
    # Calculate +DM and -DM
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    tr14 = wilders_smoothing(tr, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Parabolic SAR
    # Start with assumption of uptrend
    sar = np.full_like(high, np.nan)
    trend = np.ones_like(high, dtype=int)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = np.zeros_like(high)  # extreme point
    
    # Initialize
    sar[0] = low[0]
    trend[0] = 1
    ep[0] = high[0]
    
    for i in range(1, len(high)):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
            # Reverse if price < SAR
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = low[i]
                af = 0.02
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af = min(af + 0.02, max_af)
                else:
                    ep[i] = ep[i-1]
                    af = min(af + 0.02, max_af)
                sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
            # Reverse if price > SAR
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = high[i]
                af = 0.02
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af = min(af + 0.02, max_af)
                else:
                    ep[i] = ep[i-1]
                    af = min(af + 0.02, max_af)
                sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(sar[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: Strong uptrend + price above SAR + price above weekly pivot
            if strong_trend and price > sar[i] and price > pivot_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong downtrend + price below SAR + price below weekly pivot
            elif strong_trend and price < sar[i] and price < pivot_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Weak trend or price crosses below SAR or price below weekly S1
            if adx[i] < 20 or price < sar[i] or price < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Weak trend or price crosses above SAR or price above weekly R1
            if adx[i] < 20 or price > sar[i] or price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals