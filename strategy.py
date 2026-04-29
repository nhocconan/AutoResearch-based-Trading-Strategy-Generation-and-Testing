#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation
# Long when: price breaks above Donchian(20) upper band AND close > 1d EMA34 AND volume > 1.5x 20-period average volume
# Short when: price breaks below Donchian(20) lower band AND close < 1d EMA34 AND volume > 1.5x 20-period average volume
# Uses discrete sizing (0.25) to minimize fee churn. Donchian provides structure, EMA34 filters trend direction,
# volume spike confirms institutional interest. Works in bull/bear via trend filter.
# Timeframe: 12h (primary), HTF: 1d for EMA34.

name = "12h_Donchian20_EMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h data
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Calculate volume spike: volume > 1.5x 20-period average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if EMA34 data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band OR close crosses below EMA34
            if (curr_low < curr_donchian_lower or
                (i > 0 and close[i-1] >= ema_34_1d_aligned[i-1] and curr_close < curr_ema_34)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band OR close crosses above EMA34
            if (curr_high > curr_donchian_upper or
                (i > 0 and close[i-1] <= ema_34_1d_aligned[i-1] and curr_close > curr_ema_34)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND close > EMA34 AND volume spike
            if (curr_high > curr_donchian_upper and
                curr_close > curr_ema_34 and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band AND close < EMA34 AND volume spike
            elif (curr_low < curr_donchian_lower and
                  curr_close < curr_ema_34 and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals