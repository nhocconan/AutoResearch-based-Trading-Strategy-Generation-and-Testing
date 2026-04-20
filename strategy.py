#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Channel Breakout + 1w EMA Trend + Volume Confirmation
# - Long when price breaks above Donchian(20) high and 1w EMA(20) rising
# - Short when price breaks below Donchian(20) low and 1w EMA(20) falling
# - Volume filter: require volume > 1.5x 20-period average
# - Uses weekly EMA for trend filter to avoid whipsaws in ranging markets
# - Designed for 4h timeframe with selective entries to limit trade frequency
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(20) on 1w timeframe
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 4h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian Channel (20) on 4h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for filter
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or \
           np.isnan(volume_avg[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        donch_high = highest_high_20[i]
        donch_low = lowest_low_20[i]
        vol_average = volume_avg[i]
        ema_trend = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume confirmation + EMA up
            if price > donch_high and vol > 1.5 * vol_average and ema_trend > ema_20_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume confirmation + EMA down
            elif price < donch_low and vol > 1.5 * vol_average and ema_trend < ema_20_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or EMA turns down
            if price < donch_low or ema_trend < ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or EMA turns up
            if price > donch_high or ema_trend > ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1wEMA_VolumeFilter"
timeframe = "4h"
leverage = 1.0