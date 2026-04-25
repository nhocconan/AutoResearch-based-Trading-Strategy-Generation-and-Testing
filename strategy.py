#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Daily EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong directional moves. 
Filtering by daily EMA trend ensures alignment with higher timeframe momentum.
Volume confirmation avoids false breakouts. Works in both bull and bear markets
by taking breakouts in direction of daily trend (long in uptrend, short in downtrend).
Targets 20-50 trades/year to minimize fee drag.
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
    
    # Get 1d data for EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 4h volume spike
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian band, above daily EMA, volume confirmation
            long_entry = (curr_close > upper_band and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below lower Donchian band, below daily EMA, volume confirmation
            short_entry = (curr_close < lower_band and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below lower Donchian band OR below daily EMA
            if curr_close < lower_band or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian band OR above daily EMA
            if curr_close > upper_band or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_DailyEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0