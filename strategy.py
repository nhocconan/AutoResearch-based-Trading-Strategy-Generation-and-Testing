#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long when: price breaks above Donchian upper (20) AND price > 1d EMA50 AND volume > 2.0x avg
# Short when: price breaks below Donchian lower (20) AND price < 1d EMA50 AND volume > 2.0x avg
# Exit when: price reverts to Donchian midpoint (10-period average of high/low)
# Uses discrete sizing (0.30) to balance return and drawdown. Works in bull/bear via 1d trend filter.
# Timeframe: 4h (primary), HTF: 1d for EMA50 trend.

name = "4h_Donchian20_1dEMA50_VolumeBreakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_donchian_mid = donchian_mid[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit when price reverts to Donchian midpoint
            if curr_close <= curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit when price reverts to Donchian midpoint
            if curr_close >= curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper AND price > 1d EMA50 AND volume confirm
            if (curr_high > curr_donchian_high and
                curr_close > curr_ema_50_1d and
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian lower AND price < 1d EMA50 AND volume confirm
            elif (curr_low < curr_donchian_low and
                  curr_close < curr_ema_50_1d and
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals