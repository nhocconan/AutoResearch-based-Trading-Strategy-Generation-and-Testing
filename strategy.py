#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses weekly EMA50 for trend filter (bullish when price > EMA50, bearish when price < EMA50)
# 1d Donchian breakout: long when price breaks above 20-day high, short when breaks below 20-day low
# Volume confirmation: current volume > 1.5x 20-day average volume
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in bull/bear: EMA50 adapts to trend, Donchian provides structure with volatility adjustment

name = "1d_1w_donchian_ema_volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 with proper min_periods
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA50 to 1d timeframe
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channels on 1d data
    # Donchian upper = highest high over 20 periods, lower = lowest low over 20 periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish (price < weekly EMA50)
            if close[i] < donchian_low[i] or close[i] < ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish (price > weekly EMA50)
            if close[i] > donchian_high[i] or close[i] > ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian high AND price > weekly EMA50 (bullish trend)
                if close[i] > donchian_high[i] and close[i] > ema_50_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low AND price < weekly EMA50 (bearish trend)
                elif close[i] < donchian_low[i] and close[i] < ema_50_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals