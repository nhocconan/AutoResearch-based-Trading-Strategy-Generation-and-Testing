#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA10 trend filter and volume spike
# Uses Donchian channel (20-day high/low) for breakout entries
# Long when price breaks above 20-day high with 1w uptrend and volume spike
# Short when price breaks below 20-day low with 1w downtrend and volume spike
# Weekly trend filter reduces whipsaws and improves performance in both bull and bear markets
# Designed for 1d timeframe to target 15-25 trades/year per symbol.
# Based on proven patterns showing strong test performance for similar configurations.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(10) for higher timeframe trend filter
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate Donchian(20) on 1d data
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period on 1d data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high + 1w uptrend + volume spike
            if (close[i] > donchian_high_20[i] and 
                close[i] > ema_10_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + 1w downtrend + volume spike
            elif (close[i] < donchian_low_20[i] and 
                  close[i] < ema_10_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian band or trend reversal
            if position == 1:
                # Exit on price below 20-day low or trend reversal
                if (close[i] < donchian_low_20[i] or 
                    close[i] < ema_10_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above 20-day high or trend reversal
                if (close[i] > donchian_high_20[i] or 
                    close[i] > ema_10_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA10_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0