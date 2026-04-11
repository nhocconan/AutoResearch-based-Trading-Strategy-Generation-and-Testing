#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_trend_v32
Strategy: 4h breakout with 1d trend and volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Go long when 4h closes above prior 1d's R3 with volume confirmation and 1d uptrend; go short when 4h closes below prior 1d's S3 with volume confirmation and 1d downtrend. Exit when price returns to prior 1d's pivot. Uses only completed 1d levels to avoid look-ahead. Designed for low frequency (20-50 trades/year) to minimize fee drift while capturing momentum in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v32"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h ATR for volatility filter (optional, not used here)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Close (trend filter: use prior 1d's close) ===
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_trend = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 1d Prior OHLC for Camarilla (use prior 1d's data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Prior 1d's OHLC (shifted by 1 to avoid look-ahead)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d_prev, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    # Calculate prior 1d's Camarilla levels
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166
    s3_1d = close_1d_shift - range_1d * 1.166
    
    # Align 1d Camarilla to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(close_1d_trend[i]) or
            np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 4h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price close vs prior 1d close (1d trend)
        uptrend_1d = price_close > close_1d_trend[i]
        downtrend_1d = price_close < close_1d_trend[i]
        
        # Long conditions: 4h closes above prior 1d's R3 with volume + 1d uptrend
        long_signal = volume_confirmed and (price_close > r3_1d_aligned[i]) and uptrend_1d
        
        # Short conditions: 4h closes below prior 1d's S3 with volume + 1d downtrend
        short_signal = volume_confirmed and (price_close < s3_1d_aligned[i]) and downtrend_1d
        
        # Exit when price returns to the 1d pivot (mean reversion within prior 1d's range)
        exit_long = position == 1 and price_close < pivot_1d_aligned[i]
        exit_short = position == -1 and price_close > pivot_1d_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Go long when 4h closes above prior 1d's R3 with volume confirmation and 1d uptrend; go short when 4h closes below prior 1d's S3 with volume confirmation and 1d downtrend. Exit when price returns to prior 1d's pivot. Uses only completed 1d levels to avoid look-ahead. Designed for low frequency (20-50 trades/year) to minimize fee drift while capturing momentum in both bull and bear markets by aligning with higher timeframe trend.