#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for lower frequency and reduced fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 24-period volume MA to capture institutional interest.
- Camarilla: Calculate R3 and S3 levels from prior 1d OHLC.
- Entry: Long when close breaks above R3 AND 1d EMA34 bullish AND volume spike.
         Short when close breaks below S3 AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite breakout (close < R3 for long, close > S3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
This strategy targets institutional breakouts in the direction of the daily trend,
using Camarilla levels as dynamic support/resistance and volume to confirm participation.
Works in both bull and bear markets by only taking trades aligned with the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 24-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate Camarilla levels from prior 1d OHLC
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close_val = df_1d['close'].values
    camarilla_r3 = df_1d_close_val + 1.1 * (df_1d_high - df_1d_low) / 2
    camarilla_s3 = df_1d_close_val - 1.1 * (df_1d_high - df_1d_low) / 2
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current 4h volume > 2.0 * 24-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: break above R3 AND 1d EMA34 bullish (close > EMA)
                if curr_close > r3_level and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: break below S3 AND 1d EMA34 bearish (close < EMA)
                elif curr_close < s3_level and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: close below R3 OR loss of volume confirmation
            if curr_close < r3_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above S3 OR loss of volume confirmation
            if curr_close > s3_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0