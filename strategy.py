#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter.
- Camarilla levels: R3, S3 from prior 1d session (high, low, close).
- Entry: Long when price breaks above R3 with close > R3 AND 12h EMA50 uptrend AND volume > 1.5 * 20-period average volume.
         Short when price breaks below S3 with close < S3 AND 12h EMA50 downtrend AND volume > 1.5 * 20-period average volume.
- Exit: Opposite breakout (price crosses S3 for long, R3 for short) OR trend reversal (12h EMA50 flips).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels act as magnet points; breaks indicate strong momentum.
- 12h EMA50 filters counter-trend trades in choppy markets.
- Volume confirmation ensures institutional participation.
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d session
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d high, low, close (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3, additional_delay_bars=1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 200  # Need sufficient data for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_ratio = volume[i] / (pd.Series(volume[max(0, i-19):i+1]).mean() + 1e-10)  # 4h volume ratio
        
        # Exit conditions: opposite breakout OR trend reversal
        if position != 0:
            # Exit long: price crosses below S3 OR 12h EMA50 turns down
            if position == 1:
                if curr_low < S3_aligned[i] or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above R3 OR 12h EMA50 turns up
            elif position == -1:
                if curr_high > R3_aligned[i] or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above R3 with close > R3 AND 12h EMA50 uptrend AND volume confirmation
            if (curr_high > R3_aligned[i] and curr_close > R3_aligned[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and curr_vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with close < S3 AND 12h EMA50 downtrend AND volume confirmation
            elif (curr_low < S3_aligned[i] and curr_close < S3_aligned[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and curr_vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_TrendFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0