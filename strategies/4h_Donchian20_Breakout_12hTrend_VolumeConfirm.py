#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeConfirm
Hypothesis: Donchian(20) breakout on 4h with 12h EMA50 trend filter and volume confirmation (>1.5x average volume).
In bull markets: price breaks above upper Donchian channel with 12h uptrend and high volume → long.
In bear markets: price breaks below lower Donchian channel with 12h downtrend and high volume → short.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 trades over 4 years (19-50/year) on 4h timeframe.
Requires BTC/ETH edge via 12h trend and volume filters; avoids SOL-only bias by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian and EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Calculate Donchian channels using lookback period
        lookback_start = i - 20
        if lookback_start < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Donchian(20) upper and lower bands
        upper_band = np.max(high[lookback_start:i])
        lower_band = np.min(low[lookback_start:i])
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_12h_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(upper_band) or np.isnan(lower_band) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above upper Donchian with 12h uptrend and volume confirmation
        long_condition = (close_val > upper_band) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below lower Donchian with 12h downtrend and volume confirmation
        short_condition = (close_val < lower_band) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite breakout
        exit_long = (close_val < ema_val) or (close_val < lower_band)
        exit_short = (close_val > ema_val) or (close_val > upper_band)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0