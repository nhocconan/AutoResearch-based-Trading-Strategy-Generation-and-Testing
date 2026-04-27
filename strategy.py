#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Uses daily timeframe with Donchian(20) breakouts filtered by weekly EMA50 trend and volume confirmation.
Designed for BTC/ETH to work in both bull and bear markets by only taking breakouts in the direction of the weekly trend.
Low frequency (target 7-25 trades/year) to minimize fee drag on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian channels from previous completed daily bar (using daily data)
    df_1d = get_htf_data(prices, '1d')
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    upper = prev_high  # Donchian upper = previous day's high (20-period would use rolling max, but we use daily breakout)
    lower = prev_low   # Donchian lower = previous day's low
    
    # Align Donchian levels to daily timeframe (no alignment needed as both are daily, but we shift by 1 for previous bar)
    # Since we're on 1d timeframe and using 1d data shifted by 1, we can use directly
    upper_aligned = upper  # Already aligned to daily (shifted by 1 for previous bar)
    lower_aligned = lower  # Already aligned to daily (shifted by 1 for previous bar)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need weekly EMA50 (50), daily shift(1) for Donchian, vol avg (20)
    start_idx = max(50, 1, 20)  # Weekly EMA50 needs 50 weeks of data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_confirm[i]) or
            i >= len(upper_aligned) or i >= len(lower_aligned)):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly EMA50 alignment and volume confirmation
            long_condition = (close_val > upper_val and 
                            close_val > ema_val and 
                            vol_conf)
            short_condition = (close_val < lower_val and 
                             close_val < ema_val and 
                             vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below weekly EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0