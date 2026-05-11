# Solution
#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Calculate Camarilla levels for 4h: using previous day's OHLC
    # We'll compute daily OHLC first, then derive Camarilla for intraday
    # But since we are on 4h timeframe, we need to align daily levels to each 4h bar
    # Simpler: use 1d high/low/close to calculate Camarilla, then align
    # Camarilla R1 = close + (high - low) * 1.1 / 12
    # Camarilla S1 = close - (high - low) * 1.1 / 12
    # Actually, classic Camarilla uses previous day's close, high, low
    # We'll use 1d data: for each 1d bar, compute R1,S1 based on that day's OHLC
    # Then align to 4h
    
    # Calculate for each 1d bar
    # Note: we need previous day's OHLC for today's Camarilla levels
    # So we shift by 1 day
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use shifted values: today's Camarilla based on yesterday's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero
    range_hl = prev_high - prev_low
    # Where range is zero, set to small value to avoid division issues, but later we'll mask
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    R1 = prev_close + range_hl * 1.1 / 12
    S1 = prev_close - range_hl * 1.1 / 12
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for EMA and MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close > R1 (breakout) + 1d uptrend + volume confirmation
            if close[i] > R1_aligned[i] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < S1 (breakdown) + 1d downtrend + volume confirmation
            elif close[i] < S1_aligned[i] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close < S1 (breakdown of support) OR 1d trend turns down
            if close[i] < S1_aligned[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close > R1 (breakout of resistance) OR 1d trend turns up
            if close[i] > R1_aligned[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals