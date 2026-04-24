#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channel: 20-period high/low from prior 1d candles to avoid look-ahead.
- Entry: Long when price breaks above prior 20-day high AND 1w EMA50 bullish AND volume > 1.5 * volume MA(50).
         Short when price breaks below prior 20-day low AND 1w EMA50 bearish AND volume > 1.5 * volume MA(50).
- Exit: Close-based reversal - exit long when price crosses below prior 10-day low,
        exit short when price crosses above prior 10-day high.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures medium-term breakouts aligned with the weekly trend, designed to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate prior 20-day Donchian high/low (using prior 1d candles to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) - 20-period high/low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Prior 10-day high/low for exit (using prior 1d candles)
    exit_high = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    exit_low = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    exit_high_aligned = align_htf_to_ltf(prices, df_1d, exit_high)
    exit_low_aligned = align_htf_to_ltf(prices, df_1d, exit_low)
    
    # Calculate volume MA(50) for confirmation (using 1d data)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 60)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(exit_high_aligned[i]) or
            np.isnan(exit_low_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma_aligned[i]
            
            # Long: Price breaks above prior 20-day high AND 1w EMA50 bullish AND volume confirmed
            if curr_close > donchian_high_aligned[i] and curr_close > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 20-day low AND 1w EMA50 bearish AND volume confirmed
            elif curr_close < donchian_low_aligned[i] and curr_close < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 10-day low
            if curr_close < exit_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 10-day high
            if curr_close > exit_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0