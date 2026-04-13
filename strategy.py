#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
    # Long: price breaks above 20-day high AND price > weekly EMA50 AND volume > 1.5x 20-day avg
    # Short: price breaks below 20-day low AND price < weekly EMA50 AND volume > 1.5x 20-day avg
    # Exit: price returns to 10-day midpoint or volume drops below average
    # Using 1d/1w for signal structure, with volume confirmation to reduce false breakouts
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 20-60 trades/year on 1d timeframe (~80-240 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - reduced session to avoid overtrading
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (based on previous 20 days)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20): upper = max(high of last 20 days), lower = min(low of last 20 days)
    # Using rolling window with min_periods
    high_ma = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 1d timeframe (no alignment needed for same timeframe)
    donch_high = high_ma  # Already aligned to 1d bars
    donch_low = low_ma
    
    # Get weekly data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # Fallback to just 1d if 1w not enough data
        ema_1w = np.full(len(close_1d), np.nan)
    else:
        close_1w = df_1w['close'].values
        ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 1d timeframe (wait for completed weekly bar)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w) if len(df_1w) >= 50 else np.full(len(prices), np.nan)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period: need at least 20 days for Donchian + 50 for weekly EMA
    start_idx = max(100, 70)  # Ensure we have enough history
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC (reduced to avoid Asian session noise)
        if not in_session[i]:
            # Force flat outside session
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > weekly EMA50, only short if price < weekly EMA50
        long_trend_ok = close[i] > ema_1w_aligned[i]
        short_trend_ok = close[i] < ema_1w_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend + session
        # Note: using strict breakout (close outside channel) to avoid whipsaws
        long_entry = (close[i] > donch_high[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < donch_low[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to midpoint (10-day average of channel) or volume dry-up
        # Midpoint = (donch_high + donch_low) / 2
        midpoint = (donch_high[i] + donch_low[i]) / 2
        long_exit = (close[i] < midpoint) or not vol_confirm
        short_exit = (close[i] > midpoint) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_weekly_trend_volume_session_v1"
timeframe = "1d"
leverage = 1.0