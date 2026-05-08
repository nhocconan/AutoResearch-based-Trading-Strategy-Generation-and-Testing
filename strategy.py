#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 level touch with 1d EMA34 trend filter and volume spike
# Long when price touches S3 support + price > 1d EMA34 + volume > 2x 20-period EMA of volume
# Short when price touches R3 resistance + price < 1d EMA34 + volume > 2x 20-period EMA of volume
# Camarilla levels provide institutional support/resistance, EMA34 filters counter-trend trades, volume confirms institutional interest
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years)

name = "4h_Camarilla_R3S3_Touch_1dEMA34_Volume"
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
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_high = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else df_1d['high'].iloc[-1]
        prev_low = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else df_1d['low'].iloc[-1]
        prev_close = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else df_1d['close'].iloc[-1]
        
        # Camarilla formula
        range_val = prev_high - prev_low
        camarilla_high[i] = prev_close + range_val * 1.1 / 2
        camarilla_low[i] = prev_close - range_val * 1.1 / 2
        r3[i] = prev_close + range_val * 1.1 * 6 / 8
        s3[i] = prev_close - range_val * 1.1 * 6 / 8
    
    # Align Camarilla levels to 4h timeframe (they change only at 1d boundaries)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # start after first bar to have previous day data
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 2x 20-period EMA
        # Find the most recent completed 1d bar
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed 1d bar
        
        if idx_1d < 0:
            vol_filter = False
        else:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 2.0 * vol_ema_20_aligned[i]
        
        # Price proximity to Camarilla levels (within 0.1%)
        proximity = 0.001
        near_r3 = abs(close[i] - r3_aligned[i]) / r3_aligned[i] < proximity
        near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < proximity
        
        if position == 0:
            # Look for entry: Camarilla touch + trend + volume
            long_condition = near_s3 and close[i] > ema_34_aligned[i] and vol_filter
            short_condition = near_r3 and close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals