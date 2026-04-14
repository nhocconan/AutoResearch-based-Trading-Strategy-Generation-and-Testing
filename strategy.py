#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy with 4-hour Donchian(20) trend filter and daily volume confirmation.
# Uses 4-hour Donchian breakout in the direction of daily trend (price > daily EMA200).
# Volume > 2x daily average confirms institutional participation.
# Entry only during 08-20 UTC to avoid low-liquidity periods.
# Position size fixed at 0.20 to manage drawdown.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute hour filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 4h data ONCE for Donchian
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channel (20 periods)
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().shift(1).values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Load 1d data ONCE for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily EMA(200) for trend filter
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Daily average volume (20-period)
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # Fixed 20% position
    
    # Start after enough data for calculations
    start = max(200, 20)
    
    for i in range(start, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA200
        above_ema = close[i] > ema_200_aligned[i]
        below_ema = close[i] < ema_200_aligned[i]
        
        # Volume confirmation: current volume > 2x daily average
        volume_confirmed = volume[i] > 2.0 * vol_avg_1d_aligned[i]
        
        if position == 0:
            # Enter long: 4h Donchian breakout above + above daily EMA200 + volume
            if (close[i] > donch_high_aligned[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: 4h Donchian breakdown below + below daily EMA200 + volume
            elif (close[i] < donch_low_aligned[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to daily EMA200 or breaks below 4h Donchian low
            if close[i] < ema_200_aligned[i] or close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to daily EMA200 or breaks above 4h Donchian high
            if close[i] > ema_200_aligned[i] or close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Donchian_EMA200_Volume"
timeframe = "1h"
leverage = 1.0