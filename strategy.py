#!/usr/bin/env python3
# 12h_1d_Camarilla_R4_S4_Breakout_1dTrend_Volume_v3
# Hypothesis: 12h breakout of daily Camarilla R4/S4 levels with daily trend filter and volume confirmation.
# R4/S4 are stronger reversal zones than R3/S3, offering fewer but higher-quality breakouts.
# Long when price breaks above R4 in daily uptrend with volume surge (>2x avg), short when breaks below S4 in daily downtrend.
# Designed to reduce trade frequency while maintaining edge in both bull and bear markets via trend alignment.
# Added: Volatility filter using ATR to avoid whipsaw in low volatility conditions.

name = "12h_1d_Camarilla_R4_S4_Breakout_1dTrend_Volume_v3"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day will use invalid data, but will be filtered by alignment
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    range_prev = prev_high - prev_low
    # S4 = close - 1.1 * range / 2
    s4 = prev_close - 1.1 * range_prev / 2
    # R4 = close + 1.1 * range / 2
    r4 = prev_close + 1.1 * range_prev / 2
    
    # Align daily levels to 12h timeframe
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Daily EMA for trend filter (34-period)
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Align daily ATR for volatility filter
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation (20-period for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR-based volatility filter: only trade when volatility is above average
    vol_filter = atr_1d_aligned > 0.5 * pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(s4_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from daily: close > EMA = uptrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_1d_aligned[i]
        
        # Volume confirmation (2.0x average to reduce frequency)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: avoid low volatility conditions
        if not vol_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R4 in uptrend with volume
            if close[i] > r4_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S4 in downtrend with volume
            elif close[i] < s4_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close back below R4 or trend fails
                if close[i] < r4_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close back above S4 or trend fails
                if close[i] > s4_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals