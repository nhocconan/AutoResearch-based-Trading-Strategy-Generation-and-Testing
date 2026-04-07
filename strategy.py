#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout with Weekly Trend and Volume Confirmation
# Hypothesis: Donchian(20) breakouts on 6h timeframe, filtered by weekly trend direction
# and confirmed by volume spikes, capture institutional breakout moves. Works in bull
# via upward breakouts, in bear via downward breakdowns, and avoids false breakouts
# in ranging markets via volume and trend filters. Target: 20-40 trades/year.
name = "6h_donchian20_weekly_trend_volume_v1"
timeframe = "6h"
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
    
    # Donchian(20) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False).mean().values
    weekly_ema_6h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: 6h volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price closes below 20-period low or weekly trend turns bearish
            if close[i] < low_20[i] or close[i] < weekly_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above 20-period high or weekly trend turns bullish
            if close[i] > high_20[i] or close[i] > weekly_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above 20-period high with volume and bullish weekly trend
            if close[i] > high_20[i] and vol_confirm and close[i] > weekly_ema_6h[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 20-period low with volume and bearish weekly trend
            elif close[i] < low_20[i] and vol_confirm and close[i] < weekly_ema_6h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals