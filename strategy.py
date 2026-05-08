#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Uses Donchian(20) breakout on 4h, confirmed by 1d EMA50 trend and 1d volume > 1.3x 20-day EMA
# Designed for 4h timeframe to target 25-50 trades/year (100-200 total over 4 years)
# Donchian breakouts capture breakouts from consolidation, effective in both bull and bear markets

name = "4h_Donchian_Breakout_1dEMA50_Volume"
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
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_daily['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ema_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute Donchian channels (20-period)
    # Upper band: highest high over 20 periods
    # Lower band: lowest low over 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.3x 20-day EMA
        # Find the most recent completed daily bar
        idx_daily = 0
        while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
            idx_daily += 1
        idx_daily -= 1  # last completed daily bar
        
        if idx_daily < 0:
            vol_filter = False
        else:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.3 * vol_ema_20_aligned[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_upper[i-1]  # break above upper band
        short_breakout = close[i] < donchian_lower[i-1]  # break below lower band
        
        if position == 0:
            # Look for entry: Donchian breakout + daily trend + volume
            if long_breakout and close[i] > ema_50_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            elif short_breakout and close[i] < ema_50_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band (breakdown)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band (breakout)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals