#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-day high AND 1w EMA50 is rising (trend up) with volume > 1.5x 20-day avg
# - Short when price breaks below 20-day low AND 1w EMA50 is falling (trend down) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15 trades/year (60 total over 4 years) to avoid fee drag
# - 1w trend filter ensures we trade with the higher timeframe momentum
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - Volume confirmation filters out weak breakouts

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter (rising/falling)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_rising = ema_50_1w > np.roll(ema_50_1w, 1)  # current > previous
    ema_50_1w_falling = ema_50_1w < np.roll(ema_50_1w, 1)  # current < previous
    ema_50_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_rising)
    ema_50_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_falling)
    
    # Pre-compute 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_rising_aligned[i]) or np.isnan(ema_50_1w_falling_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above 20-day high, 1w trend up, volume spike
            if (close[i] > highest_high_20[i] and 
                ema_50_1w_rising_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below 20-day low, 1w trend down, volume spike
            elif (close[i] < lowest_low_20[i] and 
                  ema_50_1w_falling_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long when price breaks below 20-day low (opposite channel)
            if position == 1 and close[i] < lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            # Exit short when price breaks above 20-day high (opposite channel)
            elif position == -1 and close[i] > highest_high_20[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals