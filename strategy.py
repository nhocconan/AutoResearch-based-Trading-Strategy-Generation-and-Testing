#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Camarilla pivot levels provide institutional reference points; R3/S3 are strong breakout levels.
# Enter on breakout above R3 or below S3 with 1d volume confirmation and only in trending regimes (CHOP < 50).
# Designed for 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.
# Works in bull markets via breakout continuation and in bear markets via breakdown shorts with regime filter.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume spike and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume EMA20 for spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr1 = true_range(df_1d['high'], df_1d['low'], np.roll(df_1d['close'], 1))
    tr1[0] = df_1d['high'][0] - df_1d['low'][0]  # first TR
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Camarilla levels from previous day's OHLC
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1
    # S3 = Pivot - (H - L) * 1.1
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    pivot = (df_1d_high + df_1d_low + df_1d_close) / 3
    r3 = pivot + (df_1d_high - df_1d_low) * 1.1
    s3 = pivot - (df_1d_high - df_1d_low) * 1.1
    
    # Align Camarilla levels to 4h (use previous day's levels for today's trading)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid 1d indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ema_20_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 1d volume spike
        volume_spike = df_1d['volume'].values[min(i // 24, len(df_1d['volume'].values)-1)] > (2.0 * vol_ema_20_1d_aligned[i])
        
        # Regime filter: only trade in trending markets (CHOP < 50)
        trending_regime = chop_aligned[i] < 50
        
        if position == 0:
            # Long: breakout above R3 with volume spike and trending regime
            if close[i] > r3_aligned[i] and volume_spike and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 with volume spike and trending regime
            elif close[i] < s3_aligned[i] and volume_spike and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below pivot or loses momentum
            if close[i] < pivot[min(i // 24, len(pivot)-1)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above pivot or loses momentum
            if close[i] > pivot[min(i // 24, len(pivot)-1)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals