#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
    # Long: price breaks above 6h Donchian high(20) AND price > 1w EMA200 AND volume > 1.5x 20-period avg
    # Short: price breaks below 6h Donchian low(20) AND price < 1w EMA200 AND volume > 1.5x 20-period avg
    # Exit: price returns to 6h Donchian midpoint OR volume drops below average
    # Using 6h/1w for signal direction (Donchian breakout + weekly trend), volume for confirmation
    # Discrete position sizing (0.25) to minimize fee churn
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (based on previous 6h bar)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian high = rolling max of high over 20 periods
    # Donchian low = rolling min of low over 20 periods
    high_series = pd.Series(high_6h)
    low_series = pd.Series(low_6h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 6h Donchian levels to 15m (wait for completed 6h bar)
    dh_6h_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    dl_6h_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    dm_6h_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        # Fallback to just 6h if 1w not enough data
        ema_1w = np.full(len(close), np.nan)
    else:
        close_1w = df_1w['close'].values
        ema_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w) if len(df_1w) >= 200 else np.full(n, np.nan)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(dh_6h_aligned[i]) or np.isnan(dl_6h_aligned[i]) or 
            np.isnan(dm_6h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1w EMA200, only short if price < 1w EMA200
        long_trend_ok = True
        short_trend_ok = True
        if not np.isnan(ema_1w_aligned[i]):
            long_trend_ok = close[i] > ema_1w_aligned[i]
            short_trend_ok = close[i] < ema_1w_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > dh_6h_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < dl_6h_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to midpoint or volume dry-up
        long_exit = (close[i] < dm_6h_aligned[i]) or not vol_confirm
        short_exit = (close[i] > dm_6h_aligned[i]) or not vol_confirm
        
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

name = "6h_1w_donchian_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0