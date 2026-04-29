#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR stoploss
# Long when: price > Donchian upper (20) AND close > 1d EMA34 AND volume > 1.5 * avg volume
# Short when: price < Donchian lower (20) AND close < 1d EMA34 AND volume > 1.5 * avg volume
# Uses Donchian for structure, 1d EMA for higher-timeframe trend, volume for confirmation.
# Discrete sizing (0.25) to minimize fee churn. Works in bull/bear via EMA trend filter.
# Timeframe: 4h (primary), HTF: 1d for EMA34.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop for 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate average volume (20-period) for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if EMA data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_avg_volume = avg_volume[i]
        vol_threshold = 1.5 * curr_avg_volume if not np.isnan(curr_avg_volume) else np.inf
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below Donchian lower (20)
            # 2. Close falls below 1d EMA34 (trend change)
            if (curr_low < curr_donchian_low or
                curr_close < curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Donchian upper (20)
            # 2. Close rises above 1d EMA34 (trend change)
            if (curr_high > curr_donchian_high or
                curr_close > curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > Donchian upper AND close > 1d EMA34 AND volume confirmation
            long_entry = (curr_high > curr_donchian_high and
                          curr_close > curr_ema_34 and
                          curr_volume > vol_threshold)
                          
            # Short entry: price < Donchian lower AND close < 1d EMA34 AND volume confirmation
            short_entry = (curr_low < curr_donchian_low and
                           curr_close < curr_ema_34 and
                           curr_volume > vol_threshold)
                           
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals