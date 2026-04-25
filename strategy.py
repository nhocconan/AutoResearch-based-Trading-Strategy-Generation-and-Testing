#!/usr/bin/env python3
"""
1h EMA Crossover with 4h Donchian Trend Filter and Volume Spike Confirmation
Hypothesis: In trending markets, EMA(9)/EMA(21) crossovers on 1h capture momentum, 
but only when aligned with 4h Donchian(20) breakout direction and volume > 1.5x 20-bar MA.
Discrete position sizing (0.20) limits fee drag. Session filter (08-20 UTC) reduces noise.
Designed for 1h timeframe to target 60-150 trades over 4 years.
Works in bull markets via upside alignments and bear markets via downside alignments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid per-bar datetime ops
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = pd.Series(df_4h['high'])
    low_4h = pd.Series(df_4h['low'])
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1h EMA9 and EMA21 for entry timing
    close_s = pd.Series(close)
    ema_9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 20-period volume MA for volume spike confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA21, Donchian, and volume MA
    start_idx = max(21, 20)  # 21 for EMA21, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_9_val = ema_9[i]
        ema_21_val = ema_21[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # EMA crossover signals
        ema_bullish = ema_9_val > ema_21_val
        ema_bearish = ema_9_val < ema_21_val
        
        # Donchian breakout direction
        price_above_donch_high = curr_close > donch_high
        price_below_donch_low = curr_close < donch_low
        
        if position == 0:
            # Long: EMA bullish crossover + price above 4h Donchian high + volume confirmation
            long_signal = ema_bullish and price_above_donch_high and volume_confirm
            # Short: EMA bearish crossover + price below 4h Donchian low + volume confirmation
            short_signal = ema_bearish and price_below_donch_low and volume_confirm
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: EMA bearish crossover OR price breaks below 4h Donchian low
            if ema_bearish or curr_close < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: EMA bullish crossover OR price breaks above 4h Donchian high
            if ema_bullish or curr_close > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA_Crossover_4hDonchian_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0