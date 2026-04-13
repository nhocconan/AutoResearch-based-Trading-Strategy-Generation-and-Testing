#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
    # Long: price breaks above 20-period high AND volume > 1.5x 20-period average AND price > 1w EMA200
    # Short: price breaks below 20-period low AND volume > 1.5x 20-period average AND price < 1w EMA200
    # Exit: Donchian(10) opposite breakout or close crosses 1w EMA200
    # Using 1d for price action and volume, 1w only for trend filter
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 20-50 trades over 4 years (5-12/year) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA200 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels (based on previous 20 bars)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20) high = max(high of last 20 periods)
    donch_high_20 = np.full(len(high_1d), np.nan)
    donch_low_20 = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donch_high_20[i] = np.max(high_1d[i-20:i])
        donch_low_20[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1d Donchian(10) for exit
    donch_high_10 = np.full(len(high_1d), np.nan)
    donch_low_10 = np.full(len(low_1d), np.nan)
    
    for i in range(10, len(high_1d)):
        donch_high_10[i] = np.max(high_1d[i-10:i])
        donch_low_10[i] = np.min(low_1d[i-10:i])
    
    # Align 1d Donchian levels to 1d (no shift needed as we use completed bars)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_high_10_aligned = align_htf_to_ltf(prices, df_1d, donch_high_10)
    donch_low_10_aligned = align_htf_to_ltf(prices, df_1d, donch_low_10)
    
    # 1d Volume confirmation: >1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma[i] = np.mean(vol_1d[i-20:i])
    volume_spike_1d = vol_1d > (1.5 * vol_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike_aligned[i]
        
        # Trend filter: only long if price > 1w EMA200, only short if price < 1w EMA200
        long_trend_ok = close[i] > ema_1w_aligned[i]
        short_trend_ok = close[i] < ema_1w_aligned[i]
        
        # Entry logic: Donchian(20) breakout + volume + trend
        long_entry = (close[i] > donch_high_20_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < donch_low_20_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: Donchian(10) opposite breakout or trend reversal
        long_exit = (close[i] < donch_low_10_aligned[i]) or (close[i] < ema_1w_aligned[i])
        short_exit = (close[i] > donch_high_10_aligned[i]) or (close[i] > ema_1w_aligned[i])
        
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