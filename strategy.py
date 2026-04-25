#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Donchian(20) breakouts capture strong momentum. 1w EMA34 ensures we trade with higher timeframe trend.
Volume confirmation filters false breakouts. Works in both bull and bear markets: trend filter ensures we only trade with 1w momentum.
Target: 30-100 trades over 4 years (7-25/year). Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Calculate 20-period Donchian channels on 1d (primary timeframe)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 1d volume confirmation
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, and volume MA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_ma_1d = vol_ma_20_1d[i]
        
        # Volume confirmation: current 1d volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_1d
        
        if position == 0:
            # Look for entry signals
            # Long: Break above Donchian high AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > donchian_high_val and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below Donchian low AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < donchian_low_val and 
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
            if (curr_close < donchian_low_val or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price crosses above Donchian high OR EMA34 trend turns up
            if (curr_close > donchian_high_val or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0