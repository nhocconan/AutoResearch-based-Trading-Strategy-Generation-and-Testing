#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R3 and 1d EMA34 up and volume > 1.5x 20-period average.
# Short when price breaks below S3 and 1d EMA34 down and volume > 1.5x 20-period average.
# Exit when price returns to opposite H/L level or trend reverses.
# Uses 4h timeframe for entries, 1d for trend filter.
# Target: 20-50 trades per year to minimize fee drag.
# Works in bull markets via breakouts and bear via mean reversion at S3/R3 levels.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla levels from previous day
    # Need to group 4h bars by day
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Arrays to store daily Camarilla levels for each 4h bar
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    H = np.full(n, np.nan)
    L = np.full(n, np.nan)
    C = np.full(n, np.nan)
    
    # For each day, calculate Camarilla levels from previous day's OHLC
    for i, d in enumerate(unique_dates):
        if i == 0:
            continue  # Skip first day (no previous day)
        prev_date = unique_dates[i-1]
        # Get previous day's OHLC from 4h data
        prev_mask = (dates == prev_date)
        if not np.any(prev_mask):
            continue
        ph = high[prev_mask].max()
        pl = low[prev_mask].min()
        pc = close[prev_mask][-1]  # Last 4h bar of previous day
        
        # Camarilla levels
        R3_val = pc + 1.1 * (ph - pl) / 6
        S3_val = pc - 1.1 * (ph - pl) / 6
        
        # Apply to current day's 4h bars
        curr_mask = (dates == d)
        R3[curr_mask] = R3_val
        S3[curr_mask] = S3_val
        H[curr_mask] = ph
        L[curr_mask] = pl
        C[curr_mask] = pc
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_up = ema_34_1d > np.roll(ema_34_1d, 1)
    ema_34_1d_up = np.where(np.isnan(ema_34_1d_up), False, ema_34_1d_up)
    
    # Align 1d EMA trend to 4h
    ema_34_1d_up_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_up.astype(float))
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema_34_1d_up_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume spike and trend alignment
            # Long: price > R3, EMA up, volume spike
            if close[i] > R3[i] and ema_34_1d_up_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < S3, EMA down, volume spike
            elif close[i] < S3[i] and not ema_34_1d_up_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < S3 (opposite level) or trend reversal
            if close[i] < S3[i] or not ema_34_1d_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > R3 (opposite level) or trend reversal
            if close[i] > R3[i] or ema_34_1d_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals