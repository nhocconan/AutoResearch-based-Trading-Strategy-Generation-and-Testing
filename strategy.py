#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day trend filter (EMA34) and 4-hour Donchian breakout (20-period) with volume confirmation.
# Only enters during 08-20 UTC session. Uses a cooldown period (24 bars) after exit to prevent overtrading.
# Trend-following in bull markets, avoids false signals in bear/chop via EMA34 filter and volume spike requirement.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag and improve robustness.
name = "4h_1d_EMA34_Donchian20_Volume_Cooldown"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Donchian20 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 20-period high/low
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent overtrading
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Decrease cooldown counter
        if cooldown > 0:
            cooldown -= 1
        
        # Skip if any required data is NaN, outside session, or in cooldown
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20_4h_aligned[i]) or 
            np.isnan(low_20_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i] or cooldown > 0):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 AND breaks 4h Donchian high with volume
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > high_20_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 AND breaks 4h Donchian low with volume
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < low_20_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA34 or 4h Donchian low
            if close[i] < ema_34_1d_aligned[i] or close[i] < low_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 24  # 24 * 4h = 4 days cooldown
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA34 or 4h Donchian high
            if close[i] > ema_34_1d_aligned[i] or close[i] > high_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 24  # 24 * 4h = 4 days cooldown
            else:
                signals[i] = -0.25
    
    return signals