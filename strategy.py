#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day EMA trend filter and volume confirmation
# Uses 1d for trend and volume filters, 12h only for breakout detection
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in bull/bear via trend filter and volatility-based position sizing

name = "12h_Donchian_20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema_daily_34 = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume average for volume confirmation
    daily_volume = df_daily['volume'].values
    vol_ma_20 = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily EMA and volume to 12h timeframe
    ema_daily_34_aligned = align_htf_to_ltf(prices, df_daily, ema_daily_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    # Calculate 12h Donchian channel (20-period high/low)
    # Need at least 20 periods for Donchian
    if n < 20:
        return np.zeros(n)
    
    # Pre-calculate rolling max/min for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute session filter (08-20 UTC) - though less critical for 12h
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session (optional for 12h)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(ema_daily_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find the most recent completed daily bar for EMA/volume values
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: daily EMA34 direction (using previous bar to avoid look-ahead)
        ema_now = ema_daily_34_aligned[i]
        ema_prev = ema_daily_34_aligned[i-1]
        ema_uptrend = ema_now > ema_prev
        ema_downtrend = ema_now < ema_prev
        
        # Volume filter: current daily volume > 1.3x 20-day EMA
        vol_daily_current = df_daily.iloc[idx_daily]['volume']
        vol_filter = vol_daily_current > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for Donchian breakout with volume and trend confirmation
            if high[i] > donchian_high[i] and vol_filter and ema_uptrend:
                signals[i] = 0.25
                position = 1
            elif low[i] < donchian_low[i] and vol_filter and ema_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend reverses
            if low[i] <= donchian_low[i] or not ema_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend reverses
            if high[i] >= donchian_high[i] or not ema_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals