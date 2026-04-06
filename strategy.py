#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation
Hypothesis: Price reversals at Camarilla pivot levels (calculated from 1d range),
filtered by 1d trend direction (price vs EMA50), and confirmed by volume spikes,
capture mean-reversion in ranging markets and avoid counter-trend trades in strong trends.
Works in bull/bear: In uptrends, take longs at S1/S2; in downtrends, take shorts at R1/R2.
Volume confirms institutional interest at these key levels. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    r1_1d = close_1d + range_1d * 1.1 / 12
    r2_1d = close_1d + range_1d * 1.1 / 6
    r3_1d = close_1d + range_1d * 1.1 / 4
    r4_1d = close_1d + range_1d * 1.1 / 2
    
    s1_1d = close_1d - range_1d * 1.1 / 12
    s2_1d = close_1d - range_1d * 1.1 / 6
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d trend filter: EMA50
    def ema(series, period):
        result = np.full_like(series, np.nan)
        if len(series) >= period:
            multiplier = 2 / (period + 1)
            result[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                result[i] = (series[i] * multiplier) + (result[i-1] * (1 - multiplier))
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Determine trend: 1 if close > EMA50 (bullish), -1 if close < EMA50 (bearish)
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 1.5x average over last 30 periods
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 30)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S1 (profit target) OR trend turns bearish OR stoploss hit
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] <= s1_1d_aligned[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R1 (profit target) OR trend turns bullish OR stoploss hit
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] >= r1_1d_aligned[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for reversal entries at Camarilla levels
            # Long: price touches/bounces off S2/S3/S4 in bullish 1d trend with volume
            if (trend_1d_aligned[i] == 1 and volume_filter):
                if low[i] <= s2_1d_aligned[i] and close[i] > s2_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif low[i] <= s3_1d_aligned[i] and close[i] > s3_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif low[i] <= s4_1d_aligned[i] and close[i] > s4_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            # Short: price touches/bounces off R2/R3/R4 in bearish 1d trend with volume
            elif (trend_1d_aligned[i] == -1 and volume_filter):
                if high[i] >= r2_1d_aligned[i] and close[i] < r2_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                elif high[i] >= r3_1d_aligned[i] and close[i] < r3_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                elif high[i] >= r4_1d_aligned[i] and close[i] < r4_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals