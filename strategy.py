#!/usr/bin/env python3
# 1h_4h_1d_camarilla_breakout_v1
# Strategy: 1h Camarilla pivot breakout with 4h/1d trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as intraday support/resistance. Breakouts above H3 or below L3
# with 4h/1d trend alignment and volume confirmation capture high-probability moves.
# Uses 4h EMA for trend direction and 1d volume filter to avoid low-liquidity noise.
# Designed for ~15-35 trades/year to minimize fee drag in challenging 1h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate Camarilla levels for each 1h bar using previous day's OHLC
    # We need daily OHLC - resample 1h data to daily using proper alignment
    # Since we can't resample, we'll use the 1d data we already loaded
    # For each 1h bar, use the most recent completed 1d bar's OHLC
    # We'll shift the 1d data by 1 to avoid look-ahead (use previous day's data)
    if len(df_1d) >= 2:
        prev_day_high = df_1d['high'].shift(1).values  # Previous day's high
        prev_day_low = df_1d['low'].shift(1).values    # Previous day's low
        prev_day_close = df_1d['close'].shift(1).values # Previous day's close
        
        # Align to 1h timeframe
        prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
        prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
        prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
        
        # Calculate Camarilla levels
        range_val = prev_day_high_aligned - prev_day_low_aligned
        # Avoid division by zero
        range_val = np.where(range_val == 0, 1e-10, range_val)
        
        # Camarilla formulas
        H3 = prev_day_close_aligned + (range_val * 1.1 / 6)
        L3 = prev_day_close_aligned - (range_val * 1.1 / 6)
        H4 = prev_day_close_aligned + (range_val * 1.1 / 2)
        L4 = prev_day_close_aligned - (range_val * 1.1 / 2)
    else:
        # Not enough data
        H3 = L3 = H4 = L4 = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(vol_1d_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(H4[i]) or np.isnan(L4[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period 1d average
        # Note: Comparing 1h volume to 1d average - this is intentional to detect unusual activity
        vol_confirm = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: 4h EMA direction
        # For long: price above 4h EMA (uptrend)
        # For short: price below 4h EMA (downtrend)
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        # Long: Break above H3 with uptrend and volume confirmation
        if high[i] > H3[i] and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Break below L3 with downtrend and volume confirmation
        elif low[i] < L3[i] and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions
        # Long exit: Break below L3 (failed breakout) or reverse signal
        elif position == 1 and (low[i] < L3[i] or (high[i] > H4[i] and not uptrend)):
            position = 0
            signals[i] = 0.0
        # Short exit: Break above H3 (failed breakout) or reverse signal
        elif position == -1 and (high[i] > H3[i] or (low[i] < L4[i] and not downtrend)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals