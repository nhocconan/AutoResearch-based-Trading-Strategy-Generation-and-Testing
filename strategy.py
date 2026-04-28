#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# 1w EMA50 provides strong primary trend filter to avoid counter-trend trades.
# 6h Donchian(20) breakouts capture institutional price levels with proven edge.
# 1d volume spike (>2.0x 20-bar average) confirms breakout strength and reduces false signals.
# Position size 0.25 balances return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.

name = "6h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend filter and 1d data for volume spike
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Donchian(20) channels
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(high_roll_20[i]) or
            np.isnan(low_roll_20[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_roll_20[i]
        short_breakout = close[i] < low_roll_20[i]
        
        # Volume confirmation
        vol_confirm = volume_spike_1d_aligned[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Donchian level (mean reversion at channel)
        long_exit = close[i] < low_roll_20[i]   # Exit long at lower Donchian
        short_exit = close[i] > high_roll_20[i]  # Exit short at upper Donchian
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals