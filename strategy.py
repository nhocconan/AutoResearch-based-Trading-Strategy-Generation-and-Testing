#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
    # Long: price breaks above Donchian upper (20-period high) AND volume > 1.5x 20-period average AND price > 1w EMA50
    # Short: price breaks below Donchian lower (20-period low) AND volume > 1.5x 20-period average AND price < 1w EMA50
    # Exit: price returns to Donchian midpoint (mean reversion)
    # Using 1w for EMA50 (trend filter) and 1d for Donchian channels and volume
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 7-25 trades/year (~30-100 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (based on previous 20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Upper = rolling max(high, 20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian Lower = rolling min(low, 20)
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian Midpoint = (Upper + Lower) / 2
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align 1d Donchian levels to 1d (no alignment needed as same timeframe)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    donchian_mid_aligned = donchian_mid
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d (wait for completed 1w bar)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1w EMA50, only short if price < 1w EMA50
        long_trend_ok = close[i] > ema_1w_aligned[i]
        short_trend_ok = close[i] < ema_1w_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > donchian_upper_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < donchian_lower_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to Donchian midpoint (mean reversion)
        long_exit = close[i] < donchian_mid_aligned[i]
        short_exit = close[i] > donchian_mid_aligned[i]
        
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

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0