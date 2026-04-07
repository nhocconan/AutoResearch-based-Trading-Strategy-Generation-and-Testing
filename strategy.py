#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from daily timeframe with volume confirmation.
Long when price retraces to S3 level (1.125 * (Close - Low) + Low) during daily uptrend with volume > 1.5x average.
Short when price retraces to R3 level (High - 1.125 * (High - Close)) during daily downtrend with volume > 1.5x average.
Exit when price reaches the opposite pivot level (S1 for longs, R1 for shorts).
Designed for 12-37 trades/year to minimize fee freight while capturing institutional reversal points.
Works in both bull/bear markets as Camarilla levels adapt to volatility and daily trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = High - 1.125 * (High - Close)
    # S3 = Low + 1.125 * (Close - Low)
    camarilla_r3 = high_1d - 1.125 * (high_1d - close_1d)
    camarilla_s3 = low_1d + 1.125 * (close_1d - low_1d)
    camarilla_r1 = close_1d + 1.125 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.125 * (high_1d - low_1d) / 12
    
    # Align to 12h timeframe (shifted by 1 day for completed bars only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate daily trend using EMA(20) slope
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    daily_trend_up = np.zeros(len(ema_20_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_20_1d_aligned), dtype=bool)
    for i in range(1, len(ema_20_1d_aligned)):
        if not np.isnan(ema_20_1d_aligned[i]) and not np.isnan(ema_20_1d_aligned[i-1]):
            daily_trend_up[i] = ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]
            daily_trend_down[i] = ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]
    
    # Volume filter: 20-period average on 12h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 level (take profit)
            if close[i] >= camarilla_s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 level (take profit)
            if close[i] <= camarilla_r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price touches S3 level during daily uptrend
                if (close[i] <= camarilla_s3_aligned[i] * 1.001 and  # Allow small buffer
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R3 level during daily downtrend
                elif (close[i] >= camarilla_r3_aligned[i] * 0.999 and  # Allow small buffer
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals