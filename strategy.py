#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20-day high/low) with 1w trend filter and volume confirmation.
# Enter long when price breaks above 20-day high, volume > 1.5x 20-day average, and 1w EMA(21) uptrend.
# Enter short when price breaks below 20-day low, volume > 1.5x 20-day average, and 1w EMA(21) downtrend.
# Exit on opposite breakout or when price crosses 20-day EMA.
# Designed for 15-25 trades/year on 1d timeframe with focus on major trend moves.
# Volume filter ensures breakouts have conviction, reducing false signals.
# 1w trend filter prevents counter-trend trading in choppy markets.

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need enough data for 20-day high/low and EMA
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day volume average for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-day EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after 20-period lookback
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-day average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 1w trend direction
        is_uptrend = close[i] > ema_21_1w_aligned[i]
        is_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Breakout conditions
        breakout_high = high[i] > high_20[i-1]  # Break above previous 20-day high
        breakout_low = low[i] < low_20[i-1]     # Break below previous 20-day low
        
        # Entry conditions
        bullish_entry = breakout_high and vol_filter and is_uptrend
        bearish_entry = breakout_low and vol_filter and is_downtrend
        
        # Exit conditions: opposite breakout or EMA cross
        exit_long = (low[i] < low_20[i-1]) or (close[i] < ema_20[i])
        exit_short = (high[i] > high_20[i-1]) or (close[i] > ema_20[i])
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals