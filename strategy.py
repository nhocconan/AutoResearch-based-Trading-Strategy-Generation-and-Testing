#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation
# Long when: price breaks above Donchian(20) high AND 12h EMA50 rising AND volume > 1.5x 20-period avg volume
# Short when: price breaks below Donchian(20) low AND 12h EMA50 falling AND volume > 1.5x 20-period avg volume
# Uses discrete sizing (0.25) to minimize fee churn. Donchian provides structure, EMA50 trend filter avoids counter-trend trades,
# volume spike confirms institutional interest. Works in bull/bear via trend filter + breakout logic.
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
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) on 4h data
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate volume spike: volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 50)  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if HTF EMA data not available
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_ema_50_12h_prev = ema_50_12h_aligned[i-1] if i > 0 else curr_ema_50_12h
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below Donchian(20) low
            # 2. 12h EMA50 starts falling (trend change)
            if (curr_low < curr_donchian_low or
                curr_ema_50_12h < curr_ema_50_12h_prev):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Donchian(20) high
            # 2. 12h EMA50 starts rising (trend change)
            if (curr_high > curr_donchian_high or
                curr_ema_50_12h > curr_ema_50_12h_prev):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND 12h EMA50 rising AND volume spike
            if (curr_high > curr_donchian_high and
                curr_ema_50_12h > curr_ema_50_12h_prev and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND 12h EMA50 falling AND volume spike
            elif (curr_low < curr_donchian_low and
                  curr_ema_50_12h < curr_ema_50_12h_prev and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals