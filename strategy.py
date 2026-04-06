#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND 1w EMA21 up AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND 1w EMA21 down AND volume > 1.5x average
# Exit when price crosses opposite Donchian(10) level or trend reverses
# Uses weekly trend to avoid counter-trend trades in strong trends
# Target: 30-100 trades over 4 years by requiring multiple confluence factors

name = "1d_donchian_1w_trend_vol_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period for entry, 10-period for exit)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max()
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min()
    
    # 1-week trend filter (EMA21)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_21 = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean()
    ema_21_prev = np.roll(ema_21, 1)
    ema_21_prev[0] = ema_21[0]
    ema_21_up = ema_21 > ema_21_prev  # trending up
    ema_21_down = ema_21 < ema_21_prev  # trending down
    
    # Align weekly EMA trend to daily
    ema_21_up_aligned = align_htf_to_ltf(prices, df_1w, ema_21_up)
    ema_21_down_aligned = align_htf_to_ltf(prices, df_1w, ema_21_down)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(ema_21_up_aligned[i]) or np.isnan(ema_21_down_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian(10) low OR weekly trend turns down
            if close[i] <= lowest_low_10[i] or ema_21_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian(10) high OR weekly trend turns up
            if close[i] >= highest_high_10[i] or ema_21_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend + volume confirmation
            # Long: break above Donchian(20) high + weekly trend up + volume confirmation
            if (close[i] > highest_high_20[i] and ema_21_up_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian(20) low + weekly trend down + volume confirmation
            elif (close[i] < lowest_low_20[i] and ema_21_down_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals