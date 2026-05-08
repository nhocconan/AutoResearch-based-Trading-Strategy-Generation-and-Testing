#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Camarilla pivot level breakout with 4-hour EMA trend filter and volume confirmation
# Uses 4h/1d for signal direction, 1h only for entry timing to reduce false breakouts
# Session filter (08-20 UTC) applied to avoid low-liquidity periods
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# Works in bull/bear via trend filter and volatility-based position sizing

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Volume"
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
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume average for volume confirmation
    daily_volume = df_daily['volume'].values
    vol_ma_20 = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA and daily volume to 1h timeframe
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find the most recent completed daily bar for Camarilla calculation
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels from previous day
        prev_high = df_daily.iloc[idx_daily]['high']
        prev_low = df_daily.iloc[idx_daily]['low']
        prev_close = df_daily.iloc[idx_daily]['close']
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        
        # Volume filter: current daily volume > 1.3x 20-day EMA
        vol_daily_current = df_daily.iloc[idx_daily]['volume']
        vol_filter = vol_daily_current > 1.3 * vol_ma_20_aligned[i]
        
        # Trend filter: 4h EMA50 direction (using previous bar to avoid look-ahead)
        ema_now = ema_4h_50_aligned[i]
        ema_prev = ema_4h_50_aligned[i-1]
        ema_uptrend = ema_now > ema_prev
        ema_downtrend = ema_now < ema_prev
        
        if position == 0:
            # Look for Camarilla breakout with volume and trend confirmation
            if high[i] > r1 and vol_filter and ema_uptrend:
                signals[i] = 0.20
                position = 1
            elif low[i] < s1 and vol_filter and ema_downtrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot or trend reverses
            if low[i] <= pivot or not ema_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to pivot or trend reverses
            if high[i] >= pivot or not ema_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals