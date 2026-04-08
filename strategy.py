#!/usr/bin/env python3
"""
1h 4h/1d Multi-Timeframe Trend with Volume Confirmation
Hypothesis: 4h Donchian breakout (20) + 1d EMA50 trend filter + volume spike + 1h entry timing.
4h provides direction, 1d confirms higher timeframe trend, volume ensures conviction, 1h catches pullbacks to reduce false breakouts.
Works in bull/bear via trend filter and volatility-adjusted position sizing. Target 15-37 trades/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_volume_confirmation_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian Channel (20-period) - directional bias
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # 1d EMA50 - higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (reduce noise)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high_4h_aligned[i]) or np.isnan(lowest_low_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_filter[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 4h breakdown OR 1d trend reversal OR volume fails
            if (close[i] <= lowest_low_4h_aligned[i] or
                close[i] < ema_50_1d_aligned[i] or
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 4h breakout OR 1d trend reversal OR volume fails
            if (close[i] >= highest_high_4h_aligned[i] or
                close[i] > ema_50_1d_aligned[i] or
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: 4h breakout + 1d uptrend + volume + pullback entry
            if (close[i] > highest_high_4h_aligned[i-1] and
                close[i] > ema_50_1d_aligned[i] and
                vol_filter[i] and
                close[i] < (highest_high_4h_aligned[i-1] + lowest_low_4h_aligned[i-1]) / 2):  # Pullback to midpoint
                position = 1
                signals[i] = 0.20
            # Short: 4h breakdown + 1d downtrend + volume + pullback entry
            elif (close[i] < lowest_low_4h_aligned[i-1] and
                  close[i] < ema_50_1d_aligned[i] and
                  vol_filter[i] and
                  close[i] > (highest_high_4h_aligned[i-1] + lowest_low_4h_aligned[i-1]) / 2):  # Pullback to midpoint
                position = -1
                signals[i] = -0.20
    
    return signals