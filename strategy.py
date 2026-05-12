#0.30
#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_TrendFilter_Volume_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly OHLC for Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-week lookback)
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily (wait for weekly close)
    donchian_high = align_htf_to_ltf(prices, df_1w, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_min)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + above weekly EMA50 + volume spike
            if (close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and vol_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian low + below weekly EMA50 + volume spike
            elif (close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and vol_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above weekly Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals