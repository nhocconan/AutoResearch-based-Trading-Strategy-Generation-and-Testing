#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum. On 1d timeframe,
a break above the 20-day high with 1w EMA50 uptrend and volume spike (>2x 20-day vol MA)
signals bullish momentum in both bull and bear markets (via short breakdowns).
Uses discrete position sizing (0.30) to limit fee drag. Targets 30-100 trades over 4 years.
"""

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
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for safety
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_ma_20 = np.full(n, np.nan)
    low_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_ma_20[i] = np.max(high[i-19:i+1])
        low_ma_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-day volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, and volume MA
    start_idx = max(51, 20)  # 51 for EMA50 (50 + 1 for safety), 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        upper_channel = high_ma_20[i]
        lower_channel = low_ma_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-day average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Breakout conditions
        breakout_above = curr_close > upper_channel
        breakout_below = curr_close < lower_channel
        
        # Trend filter: price relative to 1w EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: break above upper channel + price above 1w EMA50 + volume confirmation
            long_signal = breakout_above and price_above_ema and volume_confirm
            # Short: break below lower channel + price below 1w EMA50 + volume confirmation
            short_signal = breakout_below and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back below upper channel OR price crosses below 1w EMA50
            if curr_close < upper_channel or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back above lower channel OR price crosses above 1w EMA50
            if curr_close > lower_channel or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0