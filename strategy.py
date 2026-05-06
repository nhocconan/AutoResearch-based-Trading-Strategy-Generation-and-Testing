#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly pivot points for trend context and daily ATR-based volatility breakout
# - Uses weekly pivot points (calculated from prior week's OHLC) to determine trend bias
# - Uses daily ATR(14) to identify volatility expansion/contraction regimes
# - Enters long when price breaks above daily high with volatility expansion in bullish weekly context
# - Enters short when price breaks below daily low with volatility expansion in bearish weekly context
# - Exits when volatility contracts or price reverses to opposite daily level
# - Designed to capture breakouts after volatility contraction with weekly trend confirmation
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyPivot_DailyATR_Breakout"
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
    
    # Get daily data for pivot points and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for pivot context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing with min_periods
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    
    # ATR ratio: current ATR / 20-period SMA of ATR (volatility expansion signal)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    vol_expansion = atr_ratio > 1.5  # Volatility expansion threshold
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots for each week
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Weekly trend bias: price above/below weekly pivot
    weekly_bias = np.where(close_1w > pivot_1w, 1, -1)  # 1=bullish, -1=bearish
    
    # Align daily indicators to 6h timeframe
    high_1d_6h = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_6h = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_6h = align_htf_to_ltf(prices, df_1d, close_1d)
    vol_expansion_6h = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    # Align weekly data to 6m timeframe (using previous week's values for lookahead safety)
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w, additional_delay_bars=1)
    weekly_bias_6h = align_htf_to_ltf(prices, df_1w, weekly_bias, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_1d_6h[i]) or np.isnan(low_1d_6h[i]) or np.isnan(close_1d_6h[i]) or
            np.isnan(vol_expansion_6h[i]) or np.isnan(pivot_1w_6h[i]) or np.isnan(weekly_bias_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for volatility expansion with weekly trend alignment
            if vol_expansion_6h[i]:
                # Long: price breaks above daily high in bullish weekly context
                if weekly_bias_6h[i] == 1 and close[i] > high_1d_6h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below daily low in bearish weekly context
                elif weekly_bias_6h[i] == -1 and close[i] < low_1d_6h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: volatility contraction or price reverses to daily low
            if not vol_expansion_6h[i] or close[i] < low_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: volatility contraction or price reverses to daily high
            if not vol_expansion_6h[i] or close[i] > high_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals