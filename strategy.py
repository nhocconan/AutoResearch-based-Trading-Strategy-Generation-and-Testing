#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivots provide intraday support/resistance levels derived from previous day's range
# R3/S3 are strong breakout levels; break above R3 with uptrend = long, break below S3 with downtrend = short
# 4h EMA50 ensures alignment with medium-term trend to avoid counter-trend whipsaws
# Volume spike (2.0x 20-period average) confirms momentum validity
# Discrete sizing 0.20 minimizes fee churn. Session filter (08-20 UTC) reduces noise.
# Target: 15-30 trades/year (60-120 total over 4 years) to stay within fee-efficient range.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots using previous day's OHLC
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Extract previous day's OHLC for each bar
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    # Align daily data to 1h bars (each daily value applies to 24 consecutive 1h bars)
    daily_open_time = df_4h['open_time'].values  # Using 4h as proxy for daily alignment
    # Actually, we need to get the actual 1d open_time from df_1d
    if 'open_time' in df_1d.columns:
        daily_times = df_1d['open_time'].values
    else:
        # Fallback: create synthetic daily times if needed
        daily_times = pd.date_range(start=prices['open_time'].iloc[0], periods=len(df_1d), freq='1D').values
    
    for i in range(n):
        current_time = prices['open_time'].iloc[i]
        # Find the most recent completed day
        mask = daily_times <= current_time
        if not np.any(mask):
            continue
        latest_day_idx = np.where(mask)[0][-1]
        if latest_day_idx > 0:  # Ensure we have previous day data
            prev_day_idx = latest_day_idx - 1
            prev_day_high[i] = df_1d.iloc[prev_day_idx]['high']
            prev_day_low[i] = df_1d.iloc[prev_day_idx]['low']
            prev_day_close[i] = df_1d.iloc[prev_day_idx]['close']
    
    # Calculate Camarilla levels
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            range_val = prev_day_high[i] - prev_day_low[i]
            R3[i] = prev_day_close[i] + range_val * 1.1 / 4
            S3[i] = prev_day_close[i] - range_val * 1.1 / 4
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 1)  # warmup for volume MA and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_R3 = R3[i]
        curr_S3 = S3[i]
        curr_ema_50 = ema_50_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Camarilla R3 AND price > 4h EMA50 (uptrend)
                if curr_close > curr_R3 and curr_close > curr_ema_50:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S3 AND price < 4h EMA50 (downtrend)
                elif curr_close < curr_S3 and curr_close < curr_ema_50:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Camarilla S3 OR price drops below 4h EMA50
            if curr_close < curr_S3 or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above Camarilla R3 OR price rises above 4h EMA50
            if curr_close > curr_R3 or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals