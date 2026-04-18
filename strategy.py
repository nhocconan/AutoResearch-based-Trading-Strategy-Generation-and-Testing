#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for calculations
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Williams %R (14-period)
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = -100 * (highest_high_12h - close_12h) / (highest_high_12h - lowest_low_12h + 1e-10)
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume spike (volume > 2.0x 20-period average)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * vol_ma_12h)
    
    # Align indicators to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # Calculate 6h Donchian channels (10-period) - tighter for fewer trades
    donchian_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or
            np.isnan(williams_r_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        trade_allowed = volume_spike_12h_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price breaks above Donchian high + EMA34 uptrend
            if (trade_allowed and 
                williams_r_12h_aligned[i] < -80 and 
                close[i] > donchian_high[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price breaks below Donchian low + EMA34 downtrend
            elif (trade_allowed and 
                  williams_r_12h_aligned[i] > -20 and 
                  close[i] < donchian_low[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or price breaks below Donchian low
            if (williams_r_12h_aligned[i] > -20 or 
                close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or price breaks above Donchian high
            if (williams_r_12h_aligned[i] < -80 or 
                close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR12_EMA34_Donchian10_VolumeSpike"
timeframe = "6h"
leverage = 1.0