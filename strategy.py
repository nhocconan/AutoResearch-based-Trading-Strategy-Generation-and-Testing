#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA50 trend filter
# Uses 4h Donchian channels for breakout signals, confirmed by volume spike (>1.5x 20-period avg volume)
# Only takes breakouts in direction of 12h EMA50 trend to avoid counter-trend whipsaws
# Position size 0.25 to manage drawdown in bear markets
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag
# Works in both bull/bear: 12h EMA filter ensures we only trade with the dominant trend

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = np.full(len(df_12h), np.nan)
    multiplier = 2 / (50 + 1)
    ema_50[0] = close_12h[0]
    for i in range(1, len(df_12h)):
        ema_50[i] = (close_12h[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_4h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian lower band OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian upper band OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and trend filter
            if volume_confirm:
                # Long breakout: price closes above upper Donchian band AND 12h trend is bullish
                if close[i] > highest_high[i] and close[i] > ema_50_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower Donchian band AND 12h trend is bearish
                elif close[i] < lowest_low[i] and close[i] < ema_50_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals