#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Daily Donchian channel breakouts capture significant momentum moves.
1-week EMA34 provides higher timeframe trend filter to avoid counter-trend trades.
Volume confirmation ensures breakouts have institutional participation.
Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag.
Works in both bull and bear markets: trend filter ensures alignment with weekly momentum,
volume confirmation reduces false breakouts. Discrete position sizing (0.30) balances return and risk.
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period Donchian channels on 1d (using previous 20 days)
    # Upper = max(high of past 20 days), Lower = min(low of past 20 days)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate 20-period volume MA for 1d volume confirmation
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_ma = vol_ma_20_1d[i]
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: Break above upper DONCHIAN AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > upper and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below lower DONCHIAN AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < lower and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price crosses below midpoint of Donchian channel OR EMA34 trend turns down
            midpoint = (upper + lower) / 2.0
            if (curr_close < midpoint or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: Price crosses above midpoint of Donchian channel OR EMA34 trend turns up
            midpoint = (upper + lower) / 2.0
            if (curr_close > midpoint or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0