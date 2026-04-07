#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) breakout + 1d trend filter + volume confirmation
# Hypothesis: Donchian breakouts capture trend momentum, while the 1d EMA filter
# ensures alignment with higher timeframe trend, reducing false signals.
# Volume confirmation ensures breakouts have conviction. This strategy aims
# for 15-25 trades/year on 12h timeframe, balancing opportunity with cost.
# Works in bull markets via upside breakouts and bear markets via downside breakouts.
name = "12h_donchian20_1dtrend_volume_v1"
timeframe = "12h"
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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # Get 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(daily_ema_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band or trend turns bearish
            if close[i] < lowest_low[i] or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band or trend turns bullish
            if close[i] > highest_high[i] or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Volume confirmation: vol_ratio > 1.5
            if vol_ratio[i] > 1.5:
                # Enter long: price breaks above upper band and above 1d EMA
                if close[i] > highest_high[i] and close[i] > daily_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price breaks below lower band and below 1d EMA
                elif close[i] < lowest_low[i] and close[i] < daily_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals