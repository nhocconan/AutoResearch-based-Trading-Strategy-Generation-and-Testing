#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channels: Calculated from prior 20-period 1h high/low (breakout structure).
- Entry: Long when price breaks above prior 20-period 1h high AND 4h EMA50 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior 20-period 1h low AND 4h EMA50 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 4h EMA50,
        exit short when price crosses above 4h EMA50.
- Signal size: 0.20 discrete to minimize fee churn and manage drawdown.
Uses 4h EMA50 trend filter (instead of 1d) for better responsiveness to trend changes while filtering noise.
Session filter: 08-20 UTC to reduce off-hour noise.
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate prior 20-period 1h Donchian channels (H20, L20)
    # Using rolling window on 1h data with min_periods=20
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate volume MA(20) for confirmation (using 1h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 60)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(high_rolling[i]) or 
            np.isnan(low_rolling[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior 20-period high AND 4h EMA50 bullish AND volume confirmed
            if curr_high > high_rolling[i] and curr_close > ema_4h_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below prior 20-period low AND 4h EMA50 bearish AND volume confirmed
            elif curr_low < low_rolling[i] and curr_close < ema_4h_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price crosses below 4h EMA50 (trend change)
            if curr_close < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price crosses above 4h EMA50 (trend change)
            if curr_close > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0