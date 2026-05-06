#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1h Camarilla pivot levels for precise entry/exit, 4h EMA50 for trend alignment (reduces whipsaw)
# Volume spike (>1.8x 20-bar average) confirms breakout strength
# Discrete sizing 0.20 to minimize fee drag; target 60-150 total trades over 4 years (15-37/year)
# Session filter 08-20 UTC to avoid low-liquidity periods
# Works in both bull/bear: breakouts capture momentum, trend filter avoids counter-trend traps, volume filter ensures participation

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA50 trend filter
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Calculate 1h Camarilla pivot levels (R3, S3)
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    R3 = pivot + range_hl * 1.1 / 2
    S3 = pivot - range_hl * 1.1 / 2
    
    # Align HTF indicators to 1h timeframe (primary)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend (price > EMA50) AND volume spike
            if close[i] > R3[i] and close[i] > ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < S3 AND downtrend (price < EMA50) AND volume spike
            elif close[i] < S3[i] and close[i] < ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < S3 (reversion to mean) OR stoploss (2*ATR)
            if close[i] < S3[i] or close[i] <= prices['high'][:i+1].max() - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > R3 (reversion to mean) OR stoploss (2*ATR)
            if close[i] > R3[i] or close[i] >= prices['low'][:i+1].min() + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals