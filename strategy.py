#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when: price breaks above Donchian(20) upper band AND close > 1d EMA34 AND volume > 1.5x 20-period avg volume
# Short when: price breaks below Donchian(20) lower band AND close < 1d EMA34 AND volume > 1.5x 20-period avg volume
# Uses Donchian for structure, 1d EMA for higher-timeframe trend filter, volume for confirmation.
# Discrete sizing (0.25) to minimize fee churn. Works in bull/bear via EMA34 trend filter.
# Timeframe: 12h (primary), HTF: 1d for EMA34.

name = "12h_Donchian20_Breakout_1dEMA34_VolumeConfirm_v1"
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
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h data
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate 20-period average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20)  # warmup for Donchian and volume avg
    
    for i in range(start_idx, n):
        # Skip if EMA34 data not available
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
        
        # Handle exits
        if position == 1:  # Long position
            # Exit when price breaks below Donchian lower band
            if curr_low < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper band
            if curr_high > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirmed = curr_volume > 1.5 * curr_avg_volume
            
            # Long entry: price breaks above Donchian upper band AND close > 1d EMA34 AND volume confirmed
            if (curr_high > curr_donchian_high and 
                curr_close > curr_ema_34 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band AND close < 1d EMA34 AND volume confirmed
            elif (curr_low < curr_donchian_low and 
                  curr_close < curr_ema_34 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals