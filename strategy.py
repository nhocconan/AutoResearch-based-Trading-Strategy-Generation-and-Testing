#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# Works in bull markets by buying breakouts above 4h high in uptrend, and in bear markets by selling breakdowns below 4h low in downtrend.
# Volume filter ensures breakouts have institutional participation. Target: 25-40 trades/year via strict entry conditions.

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Volume filter: 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    # Fill beginning with first valid value
    if not np.isnan(vol_ma[19]):
        vol_ma[:19] = vol_ma[19]
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= ema_period:
        ema_daily[ema_period - 1] = np.mean(close_daily[:ema_period])
        for i in range(ema_period, len(close_daily)):
            ema_daily[i] = (close_daily[i] * 2 + ema_daily[i - 1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 4h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(donchian_period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses or price breaks below lower channel
            if downtrend_htf or close[i] < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses or price breaks above upper channel
            if uptrend_htf or close[i] > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel, uptrend on daily EMA, volume confirmation
            if (close[i] > upper_channel[i] and 
                uptrend_htf and 
                volume_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel, downtrend on daily EMA, volume confirmation
            elif (close[i] < lower_channel[i] and 
                  downtrend_htf and 
                  volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals