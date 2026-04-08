#!/usr/bin/env python3
# 1d_1w_price_channel_volume_v1
# Hypothesis: Daily price channel (Donchian 20) breakout with weekly trend filter and volume confirmation.
# Works in bull markets by buying breakouts above channel, in bear markets by selling breakdowns below channel.
# Volume filter ensures institutional participation, reducing false signals.
# Target: 10-25 trades/year via Donchian breakouts + weekly trend + volume confirmation.

name = "1d_1w_price_channel_volume_v1"
timeframe = "1d"
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
    period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Weekly trend filter (EWMA 50)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # EMA 50 on weekly
    ema_period = 50
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period-1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align weekly EMA to daily
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume filter: 20-day average volume
    vol_ma = np.full(n, np.nan)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_weekly_aligned[i]
        weekly_downtrend = close[i] < ema_weekly_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below lower channel or trend reverses
            if close[i] < lower[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above upper channel or trend reverses
            if close[i] > upper[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel, volume, weekly uptrend
            if close[i] > upper[i] and volume_filter and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel, volume, weekly downtrend
            elif close[i] < lower[i] and volume_filter and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals