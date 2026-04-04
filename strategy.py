#!/usr/bin/env python3
"""
exp_6457_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
Works in bull/bear: Donchian captures breakouts, 1d EMA filters direction, volume avoids fakeouts.
Target: 75-200 trades over 4 years (19-50/year). Discrete sizing: ±0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6457_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA(50) on HTF close
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    # Align to LTF with shift(1) for completed bars only
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period volume average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    
    # Warmup: start after Donchian and EMA warmup
    start_idx = max(50, 20)  # EMA(50) needs 50, Donchian needs 20
    
    for i in range(start_idx, n):
        # Get current values
        close_price = prices['close'].iloc[i]
        vol_current = prices['volume'].iloc[i]
        vol_average = vol_ma.iloc[i]
        
        # Skip if volume data not ready
        if pd.isna(vol_average) or vol_average == 0:
            continue
            
        # Volume confirmation: current volume > 1.5x average
        volume_ok = vol_current > 1.5 * vol_average
        
        # Donchian breakout conditions
        upper_break = close_price > high_roll.iloc[i-1]  # Break above previous upper band
        lower_break = close_price < low_roll.iloc[i-1]   # Break below previous lower band
        
        # 1d EMA trend filter
        ema_trend = ema_1d_aligned[i] if not pd.isna(ema_1d_aligned[i]) else 0
        
        # Long: price breaks above Donchian upper + above 1d EMA + volume confirmation
        if upper_break and close_price > ema_trend and volume_ok:
            signals[i] = 0.25  # Long 25%
        # Short: price breaks below Donchian lower + below 1d EMA + volume confirmation
        elif lower_break and close_price < ema_trend and volume_ok:
            signals[i] = -0.25  # Short 25%
        # Otherwise flat
        else:
            signals[i] = 0.0
    
    return signals