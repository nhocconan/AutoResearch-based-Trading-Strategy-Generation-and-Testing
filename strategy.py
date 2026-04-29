#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction + 1d EMA(50) trend filter + volume spike confirmation
# Long when price breaks above 4h Donchian high AND price > 1d EMA(50) AND volume > 2.0x 24-period average (08-20 UTC session)
# Short when price breaks below 4h Donchian low AND price < 1d EMA(50) AND volume > 2.0x 24-period average (08-20 UTC session)
# Uses discrete position sizing (0.20) to minimize fee drag. Session filter reduces noise trades.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_Donchian20_Breakout_1dEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 24)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high = donchian_high_4h_aligned[i]
        curr_donchian_low = donchian_low_4h_aligned[i]
        curr_ema = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 24-period average
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
        else:
            vol_ma_24 = 0.0
        vol_spike = curr_volume > 2.0 * vol_ma_24 if vol_ma_24 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below 4h Donchian low OR price < 1d EMA(50)
            if curr_close < curr_donchian_low or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above 4h Donchian high OR price > 1d EMA(50)
            if curr_close > curr_donchian_high or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above 4h Donchian high AND price > 1d EMA(50) AND volume spike
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low AND price < 1d EMA(50) AND volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals