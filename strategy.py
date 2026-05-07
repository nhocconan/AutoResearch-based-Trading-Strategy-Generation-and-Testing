#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation
# Long when: Price breaks above Donchian(20) high AND weekly EMA trend is up AND volume spike
# Short when: Price breaks below Donchian(20) low AND weekly EMA trend is down AND volume spike
# Exit when: Price reverses to opposite Donchian band or trend changes
# Designed for low trade frequency (target: 10-20/year) to minimize fee drift
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns

name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA trend filter (34-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 1.5 * 50-period EMA
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_spike = volume > (1.5 * vol_ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + weekly uptrend + volume spike
            long_condition = (close[i] > highest_high[i]) and (close[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: Price breaks below Donchian low + weekly downtrend + volume spike
            short_condition = (close[i] < lowest_low[i]) and (close[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below Donchian low OR weekly trend turns down
            if (close[i] < lowest_low[i]) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above Donchian high OR weekly trend turns up
            if (close[i] > highest_high[i]) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals