#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w EMA Trend Filter and Volume Confirmation
Hypothesis: Daily Donchian(20) breakouts capture medium-term trends. 
Using 1w EMA34 as higher-timeframe trend filter ensures alignment with weekly trend, reducing false signals. 
Volume confirmation (>1.5x 20-day average) adds conviction to breakouts.
Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band) by requiring trend alignment.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-30 trades/year on 1d.
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
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian band AND above 1w EMA34 AND volume confirmation
            long_entry = (curr_close > upper_band) and (curr_close > ema_trend) and volume_confirm
            # Short: price breaks below lower Donchian band AND below 1w EMA34 AND volume confirmation
            short_entry = (curr_close < lower_band) and (curr_close < ema_trend) and volume_confirm
            
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
            # Exit: price falls below lower Donchian band OR price < 1w EMA34 (trend change)
            if (curr_close < lower_band) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian band OR price > 1w EMA34 (trend change)
            if (curr_close > upper_band) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0