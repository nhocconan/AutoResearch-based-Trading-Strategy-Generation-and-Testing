#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + volume confirmation + 1w EMA50 trend filter
# Uses daily Donchian channels for breakout signals, confirmed by volume spike (>1.5x 20-day avg volume)
# Only takes breakouts in direction of weekly EMA50 trend to avoid counter-trend whipsaws
# Position size 0.25 to manage drawdown in bear markets
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Works in both bull/bear: weekly EMA filter ensures we only trade with the dominant trend

name = "1d_1w_donchian_volume_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = np.full(len(df_1w), np.nan)
    multiplier = 2 / (50 + 1)
    ema_50[0] = close_1w[0]
    for i in range(1, len(df_1w)):
        ema_50[i] = (close_1w[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 1w EMA50 to daily timeframe
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels (20-period) on daily data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-day average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian lower band OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian upper band OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and trend filter
            if volume_confirm:
                # Long breakout: price closes above upper Donchian band AND weekly trend is bullish
                if close[i] > highest_high[i] and close[i] > ema_50_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower Donchian band AND weekly trend is bearish
                elif close[i] < lowest_low[i] and close[i] < ema_50_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals