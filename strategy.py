# 12h_donchian20_1d_volume_regime_v1
# Donchian breakout strategy with volume confirmation and regime filter
# Uses daily timeframe for trend filter and 12h for entry timing
# Long: Price breaks above Donchian(20) high + above daily EMA50 + volume > average
# Short: Price breaks below Donchian(20) low + below daily EMA50 + volume > average
# Designed for low frequency (target: 20-40 trades/year) to minimize fee impact
# Works in both bull/bear via trend filter - only trades in direction of higher timeframe trend

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channels (20-period)
    # We need to calculate Donchian on 12h data, but we can use the current prices directly
    # since we're already on 12h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from daily EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > high_roll[i]
        breakout_short = close[i] < low_roll[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on reverse breakout or trend change
            if breakout_short or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse breakout or trend change
            if breakout_long or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: Donchian breakout + uptrend + volume
            if breakout_long and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakdown + downtrend + volume
            elif breakout_short and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals