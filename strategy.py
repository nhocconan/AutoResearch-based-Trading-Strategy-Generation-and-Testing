#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_v2
Strategy: 12h breakout with 1d Camarilla confluence
Timeframe: 12h
Leverage: 1.0
Hypothesis: Buy when 12h closes above 1d R3 with volume confirmation; sell when 12h closes below 1d S3 with volume confirmation. Uses 1d Camarilla for entry levels and filters with 12h trend (EMA25) to avoid counter-trend trades. Designed for low frequency (target 15-30 trades/year) to minimize fee drag and work in both bull and bear markets by aligning with intermediate trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 12h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h EMA25 for trend filter
    close_series = pd.Series(close)
    ema_25 = close_series.ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # === 1d Camarilla (entry levels from prior 1d) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior 1d's values to avoid look-ahead
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166
    s3_1d = close_1d_shift - range_1d * 1.166
    
    # Align 1d Camarilla to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Session filter: 00-23 UTC (12h captures full day, but avoid low liquidity hours if needed)
    # We'll use a simple time filter: avoid 6-hour window around 04:00-10:00 UTC (lower volatility)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    # Avoid 4-10 UTC (6 hours of lower liquidity)
    in_session = ~((hours >= 4) & (hours <= 10))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_25[i]) or np.isnan(atr_12h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 12h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price vs EMA25
        uptrend = price_close > ema_25[i]
        downtrend = price_close < ema_25[i]
        
        # Long conditions: 12h closes above prior 1d's R3 with volume + uptrend
        long_signal = volume_confirmed and (price_close > r3_1d_aligned[i]) and uptrend
        
        # Short conditions: 12h closes below prior 1d's S3 with volume + downtrend
        short_signal = volume_confirmed and (price_close < s3_1d_aligned[i]) and downtrend
        
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

# Hypothesis: Buy when 12h closes above 1d R3 with volume confirmation; sell when 12h closes below 1d S3 with volume confirmation. Uses 1d Camarilla for entry levels and filters with 12h trend (EMA25) to avoid counter-trend trades. Designed for low frequency (target 15-30 trades/year) to minimize fee drag and work in both bull and bear markets by aligning with intermediate trend.