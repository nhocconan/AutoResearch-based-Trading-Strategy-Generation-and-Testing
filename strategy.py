#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h 1-Week Donchian Breakout with 1-Week Trend Filter and Volume Confirmation
# Hypothesis: 1-week Donchian(20) breakouts with 1-week EMA50 trend filter capture major trends.
# Volume confirmation ensures institutional participation. Works in bull/bear via trend filter.
# 12h timeframe with 1-week HTF targets 12-37 trades/year (50-150 over 4 years).
name = "12h_1w_donchian20_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for Donchian and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1-week Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # 1-week EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # Align 1-week data to 12h timeframe (shifted by 1 bar for completed bars only)
    donch_high_12h = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1w, donch_low)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(150, n):
        # Skip if required data not available
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(ema_1w_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1-week Donchian low
            if close[i] < donch_low_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above 1-week Donchian high
            if close[i] > donch_high_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above 1-week Donchian high with uptrend
                if close[i] > donch_high_12h[i] and close[i] > ema_1w_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 1-week Donchian low with downtrend
                elif close[i] < donch_low_12h[i] and close[i] < ema_1w_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals