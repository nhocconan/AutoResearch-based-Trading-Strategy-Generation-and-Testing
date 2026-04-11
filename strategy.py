#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high with 12h EMA(50) uptrend and volume > 1.5x average.
# Short when price breaks below Donchian(20) low with 12h EMA(50) downtrend and volume > 1.5x average.
# Exit on opposite Donchian break or trailing stop.
# Designed for 20-50 trades/year on 4h timeframe with focus on clean breakouts.
# Volume filter ensures breakouts have conviction, reducing false signals.
# 12h trend filter prevents counter-trend trading in choppy markets.

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 12h trend direction
        is_uptrend = close[i] > ema_50_12h_aligned[i]
        is_downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        bullish_breakout = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        bearish_breakout = close[i] < donchian_low[i-1]   # Break below previous Donchian low
        
        bullish_entry = bullish_breakout and vol_filter and is_uptrend
        bearish_entry = bearish_breakout and vol_filter and is_downtrend
        
        # Exit conditions: opposite Donchian break
        exit_long = close[i] < donchian_low[i-1]   # Break below Donchian low
        exit_short = close[i] > donchian_high[i-1] # Break above Donchian high
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals