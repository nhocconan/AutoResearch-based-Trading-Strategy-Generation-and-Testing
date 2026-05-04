#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance.
# Breakouts above R3 or below S3 with higher timeframe trend alignment capture strong momentum.
# Volume spike (>1.5x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 75-200 trades over 4 years.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Map each 4h bar to its corresponding 1d bar index
    # Since prices.index is DatetimeIndex, we can use date alignment
    dates_4h = prices.index.date
    dates_1d = df_1d.index.date
    
    # Create mapping from 4h bar to 1d bar index
    date_to_1d_idx = {date: idx for idx, date in enumerate(dates_1d)}
    
    # Pre-calculate Camarilla levels for each 1d bar
    camarilla_dict = {}
    for idx in range(len(df_1d)):
        h = df_1d['high'].iloc[idx]
        l = df_1d['low'].iloc[idx]
        c = df_1d['close'].iloc[idx]
        range_hl = h - l
        camarilla_dict[idx] = {
            'H': h,
            'L': l,
            'R3': c + range_hl * 1.1 / 4,
            'S3': c - range_hl * 1.1 / 4
        }
    
    # Map Camarilla levels to each 4h bar (use previous day's levels)
    for i in range(n):
        date = dates_4h[i]
        # Get previous trading day's date
        # Simple approach: subtract 1 day and check if it exists in our mapping
        from datetime import timedelta
        prev_date = date - timedelta(days=1)
        # Skip weekends - if prev_date is Saturday/Sunday, go back further
        while prev_date.weekday() > 4:  # 5=Saturday, 6=Sunday
            prev_date -= timedelta(days=1)
        
        if prev_date in date_to_1d_idx:
            idx_1d = date_to_1d_idx[prev_date]
            camarilla_high[i] = camarilla_dict[idx_1d]['H']
            camarilla_low[i] = camarilla_dict[idx_1d]['L']
            camarilla_r3[i] = camarilla_dict[idx_1d]['R3']
            camarilla_s3[i] = camarilla_dict[idx_1d]['S3']
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H (pivot high) OR trend changes OR volume drops
            if (close[i] < camarilla_high[i] or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla L (pivot low) OR trend changes OR volume drops
            if (close[i] > camarilla_low[i] or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals