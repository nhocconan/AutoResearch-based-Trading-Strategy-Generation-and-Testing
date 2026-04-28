#102568
#!/usr/bin/env python3
"""
12h_1W_Swing_Pattern_Reversal
Hypothesis: Uses 1-week swing high/low patterns combined with 12h price action to capture reversal opportunities in BTC/ETH. The strategy looks for price rejection at weekly swing levels with volume confirmation, targeting fewer than 30 trades per year to minimize fee drag. Works in both bull and bear markets by trading reversals at key weekly levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for swing levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly swing highs and lows (using 3-bar lookback)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Swing high: current high > previous high and next high
    swing_high = np.zeros_like(high_1w, dtype=bool)
    swing_low = np.zeros_like(low_1w, dtype=bool)
    
    for i in range(1, len(high_1w)-1):
        if high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i+1]:
            swing_high[i] = True
        if low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i+1]:
            swing_low[i] = True
    
    # Get swing levels (only at swing points)
    swing_high_levels = np.where(swing_high, high_1w, np.nan)
    swing_low_levels = np.where(swing_low, low_1w, np.nan)
    
    # Forward fill to get the most recent swing level
    swing_high_series = pd.Series(swing_high_levels)
    swing_low_series = pd.Series(swing_low_levels)
    swing_high_ffill = swing_high_series.ffill().values
    swing_low_ffill = swing_low_series.ffill().values
    
    # Align swing levels to 12h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1w, swing_high_ffill)
    swing_low_aligned = align_htf_to_ltf(prices, df_1w, swing_low_ffill)
    
    # Get daily data for trend filter (optional)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on daily close for trend context
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: >1.5x 24-period MA (2 days of 12h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(swing_high_aligned[i]) or 
            np.isnan(swing_low_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Price rejection at swing levels with volume
        near_swing_high = abs(close[i] - swing_high_aligned[i]) / close[i] < 0.005  # Within 0.5%
        near_swing_low = abs(close[i] - swing_low_aligned[i]) / close[i] < 0.005   # Within 0.5%
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_24[i])
        
        # Price action: look for rejection (close near open for bears/bulls)
        body_size = abs(close[i] - prices['open'].iloc[i])
        candle_range = high[i] - low[i]
        is_doji_like = body_size / candle_range < 0.3 if candle_range > 0 else False
        
        # Rejection signals
        rejection_high = near_swing_high and is_doji_like and vol_confirm
        rejection_low = near_swing_low and is_doji_like and vol_confirm
        
        # Trend filter (optional - use daily EMA for context)
        uptrend_context = close[i] > ema_20_1d_aligned[i]
        downtrend_context = close[i] < ema_20_1d_aligned[i]
        
        # Entry logic: short at swing high rejection, long at swing low rejection
        if rejection_high and downtrend_context and position >= 0:
            signals[i] = -0.25
            position = -1
        elif rejection_low and uptrend_context and position <= 0:
            signals[i] = 0.25
            position = 1
        # Exit: when price moves back toward the middle of the swing range
        elif position == 1 and close[i] < (swing_low_aligned[i] + swing_high_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (swing_low_aligned[i] + swing_high_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1W_Swing_Pattern_Reversal"
timeframe = "12h"
leverage = 1.0