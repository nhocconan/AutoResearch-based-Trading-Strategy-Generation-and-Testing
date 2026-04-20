#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian Breakout with 4h Trend Filter and 1d Volume Spike
# - Use 4h EMA21 as trend filter (long when price > EMA21, short when price < EMA21)
# - Use 1d volume spike (volume > 2.0 * 20-day average volume) to confirm momentum
# - Enter on 1h Donchian(20) breakout in direction of 4h trend with volume confirmation
# - Exit when price crosses opposite Donchian band or trend reverses
# - Designed for 1h timeframe with selective entries to avoid overtrading
# - Target: 15-37 trades per year per symbol (60-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Load 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_avg_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate Donchian channels on 1h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(ema_21_4h_aligned[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # Convert to boolean
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + 4h uptrend + volume spike
            if price > highest_high_20[i] and price > ema_21_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below lower Donchian + 4h downtrend + volume spike
            elif price < lowest_low_20[i] and price < ema_21_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or 4h trend turns down
            if price < lowest_low_20[i] or price < ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or 4h trend turns up
            if price > highest_high_20[i] or price > ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian_4hEMA_1dVolume_Spike"
timeframe = "1h"
leverage = 1.0