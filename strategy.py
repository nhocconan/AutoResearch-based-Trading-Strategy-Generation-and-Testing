#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation
# Long when: price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 2.0x 20-period avg
# Short when: price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 2.0x 20-period avg
# Exit: opposite Donchian breakout OR price crosses 12h EMA50
# Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 12h trend filter.
# Timeframe: 4h (primary), HTF: 12h for EMA50 trend.

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 20)  # warmup for Donchian, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donch_high = high_roll_max[i]
        curr_donch_low = low_roll_min[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price breaks below Donchian(20) low (contrarian breakout)
            # 2. Price falls below 12h EMA50 (trend change)
            if (curr_low <= curr_donch_low or
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price breaks above Donchian(20) high (contrarian breakout)
            # 2. Price rises above 12h EMA50 (trend change)
            if (curr_high >= curr_donch_high or
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian(20) high AND price > 12h EMA50 AND volume spike
            if (curr_high > curr_donch_high and
                curr_close > curr_ema_50_12h and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian(20) low AND price < 12h EMA50 AND volume spike
            elif (curr_low < curr_donch_low and
                  curr_close < curr_ema_50_12h and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals