#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Donchian(20) breakouts capture strong momentum, filtered by 1d EMA34 trend and volume confirmation.
Works in both bull and bear markets: EMA34 trend ensures we trade with higher timeframe direction,
while Donchian breakouts and volume confirmation reduce false signals. Discrete position sizing (0.25) minimizes fee churn.
Target: 75-200 total trades over 4 years (19-50/year) on BTC/ETH/SOL.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(donchian_period - 1, n):
        highest_high[i] = np.max(high[i-donchian_period+1:i+1])
        lowest_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Calculate 24-period volume MA for 4h volume confirmation (24 periods = 4 days of 4h data)
    vol_ma_24_4h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24_4h[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, and volume MA
    start_idx = max(donchian_period - 1, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_4h = vol_ma_24_4h[i]
        
        # Volume confirmation: current 4h volume > 1.8 * 24-period average
        volume_confirm = curr_volume > 1.8 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: Break above Donchian high AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > donchian_high and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below Donchian low AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < donchian_low and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price crosses below Donchian low OR EMA34 trend turns down
            if (curr_close < donchian_low or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price crosses above Donchian high OR EMA34 trend turns up
            if (curr_close > donchian_high or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0