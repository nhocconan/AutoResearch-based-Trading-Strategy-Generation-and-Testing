#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period high AND 12h EMA50 uptrend AND volume spike
# Short when price breaks below 20-period low AND 12h EMA50 downtrend AND volume spike
# Exit when price crosses 20-period opposite band OR trend changes
# Donchian provides clear structure, 12h EMA50 filters higher timeframe trend,
# volume confirmation ensures momentum validity and reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        period_high = np.max(high[lookback_start:i+1])
        period_low = np.min(low[lookback_start:i+1])
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price crosses below 20-period low OR price below 12h EMA50 (trend change)
            if curr_low <= period_low or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 20-period high OR price above 12h EMA50 (trend change)
            if curr_high >= period_high or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above 20-period high AND price above 12h EMA50 AND volume spike
            if (curr_high > period_high and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period low AND price below 12h EMA50 AND volume spike
            elif (curr_low < period_low and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals