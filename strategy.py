#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_donchian_breakout_volume_v1
# Weekly Donchian breakout (20-week high/low) with daily volume confirmation
# Works in bull markets (breakouts above 20w high) and bear markets (breakdowns below 20w low)
# Volume filter ensures breakouts have institutional participation
# Low trade frequency expected (<20/year) due to weekly channel and volume filter
name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period weekly Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-week high (maximum of last 20 weekly highs)
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # 20-week low (minimum of last 20 weekly lows)
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for Donchian calculation
        # Skip if Donchian levels not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        bullish_breakout = (
            close[i] > donchian_high_aligned[i] and  # Price breaks above 20w high
            volume_filter[i]  # Volume confirmation
        )
        
        bearish_breakout = (
            close[i] < donchian_low_aligned[i] and  # Price breaks below 20w low
            volume_filter[i]  # Volume confirmation
        )
        
        # Exit conditions: opposite breakout or loss of momentum
        exit_long = close[i] < donchian_low_aligned[i]  # Price breaks below 20w low
        exit_short = close[i] > donchian_high_aligned[i]  # Price breaks above 20w high
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals