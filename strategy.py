#!/usr/bin/env python3
"""
1h Donchian Breakout with 4h EMA34 Trend and Volume Spike
Hypothesis: Donchian(20) breakouts capture trend starts. 4h EMA34 filters direction, 
volume spike confirms momentum. 1h used only for entry timing to reduce noise.
Target: 60-150 trades over 4 years (15-37/year) with session filter (08-20 UTC).
Works in bull (breakouts up) and bear (breakouts down) via trend filter.
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
    
    # Get 4h data for EMA34 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 4h close for trend
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for ATR filter (volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14_1d = np.zeros(len(close_1d))
    atr_14_1d[0] = np.nan
    for i in range(1, len(tr) + 1):
        if i < 14:
            atr_14_1d[i] = np.nan
        elif i == 14:
            atr_14_1d[i] = np.mean(tr[:14])
        else:
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i-1]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # open_time is datetime64[ms], index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 20-period Donchian channels on 1h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 1h volume spike
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and ATR
    start_idx = max(20, 20)  # 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_4h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma = vol_ma_20[i]
        atr_1d = atr_14_1d_aligned[i]
        
        # Volume confirmation: current 1h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr_1d > (curr_close * 0.01)  # ATR > 1% of price
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND price > EMA34 (uptrend) AND volume confirmation AND vol filter
            long_entry = (curr_close > upper and 
                         curr_close > ema_trend and volume_confirm and vol_filter)
            # Short: price breaks below Donchian low AND price < EMA34 (downtrend) AND volume confirmation AND vol filter
            short_entry = (curr_close < lower and 
                          curr_close < ema_trend and volume_confirm and vol_filter)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low OR price falls below EMA34
            if (curr_close < lower or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high OR price rises above EMA34
            if (curr_close > upper or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0