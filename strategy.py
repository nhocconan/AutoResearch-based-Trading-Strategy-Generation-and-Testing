#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 12h Trend + Volume Confirmation
# Hypothesis: Donchian(20) breakouts capture trend continuations, filtered by 12h EMA trend and volume spikes.
# Works in bull markets by catching breakouts, in bear markets by avoiding false breakouts via trend filter.
# Target: 20-50 trades/year to minimize fee drag on 4h timeframe.
name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12-hour EMA(50) for trend filter
    daily_close_12h = df_12h['close'].values
    daily_ema_12h = pd.Series(daily_close_12h).ewm(span=50, adjust=False).mean().values
    daily_ema_12h_4h = align_htf_to_ltf(prices, df_12h, daily_ema_12h)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(daily_ema_12h_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or trend turns bearish
            if close[i] < low_20[i] or close[i] < daily_ema_12h_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or trend turns bullish
            if close[i] > high_20[i] or close[i] > daily_ema_12h_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and breakout in direction of 12h trend
            if vol_filter[i]:
                # Long: price breaks above upper Donchian band + price above 12h EMA
                if close[i] > high_20[i] and close[i] > daily_ema_12h_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band + price below 12h EMA
                elif close[i] < low_20[i] and close[i] < daily_ema_12h_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals