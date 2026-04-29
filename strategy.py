#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper AND 1d EMA50 uptrend AND volume spike
# Short when price breaks below Donchian lower AND 1d EMA50 downtrend AND volume spike
# Exit on opposite Donchian break or trend reversal
# Donchian provides clear structure, 1d EMA50 filters for higher timeframe trend,
# volume confirmation ensures momentum validity
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    period_donchian = 20
    donchian_high = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    donchian_low = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_donchian, 50)  # Donchian20 + 1d EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR 1d EMA50 downtrend
            if curr_close < curr_donchian_low or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR 1d EMA50 uptrend
            if curr_close > curr_donchian_high or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND 1d EMA50 uptrend AND volume spike
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_1d and  # price above 1d EMA50 for uptrend
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND 1d EMA50 downtrend AND volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_1d and  # price below 1d EMA50 for downtrend
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals