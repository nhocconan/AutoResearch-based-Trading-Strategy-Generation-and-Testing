#!/usr/bin/env python3
# 12h_donchian_breakout_volume_trend_v1
# Hypothesis: 12h Donchian channel breakouts with volume confirmation and 1w trend filter.
# In bull markets: buy breakouts above upper band in uptrend.
# In bear markets: sell breakdowns below lower band in downtrend.
# Uses weekly trend to avoid counter-trend trades, volume to confirm institutional interest.
# Target: 15-30 trades/year via strict breakout conditions + trend + volume filters.

name = "12h_donchian_breakout_volume_trend_v1"
timeframe = "12h"
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
    lookback = 20
    upper_band = np.full_like(high, np.nan)
    lower_band = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, len(high)):
        upper_band[i] = np.max(high[i - lookback + 1:i + 1])
        lower_band[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: 30-period average volume
    vol_ma = np.full_like(volume, np.nan)
    for i in range(29, len(volume)):
        vol_ma[i] = np.mean(volume[i - 29:i + 1])
    
    # Get weekly data for trend filter (higher timeframe)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_weekly = np.full_like(close_weekly, np.nan)
    for i in range(ema_period - 1, len(close_weekly)):
        ema_weekly[i] = np.mean(close_weekly[i - ema_period + 1:i + 1])
    
    # Align weekly EMA to 12h timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(lookback, 29, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 30-period average
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below weekly EMA
        uptrend_htf = close[i] > ema_weekly_aligned[i]
        downtrend_htf = close[i] < ema_weekly_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price crosses below lower band or trend reverses
            if close[i] < lower_band[i] or not uptrend_htf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price crosses above upper band or trend reverses
            if close[i] > upper_band[i] or not downtrend_htf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volume and uptrend
            if (close[i] > upper_band[i] and 
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume and downtrend
            elif (close[i] < lower_band[i] and 
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals