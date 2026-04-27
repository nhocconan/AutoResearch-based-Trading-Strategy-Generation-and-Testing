# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with bullish trend and volume > 1.5x average.
# Short when price breaks below lower Donchian channel with bearish trend and volume > 1.5x average.
# Exit when price crosses the 1d EMA34 (trend reversal signal).
# Uses 12h timeframe for trading frequency and 1d for trend filter to reduce noise.
# Target: 15-35 trades/year to minimize fee drag while capturing strong trending moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12-hour Donchian channel (20-period)
    upper_donchian = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (since we're trading on 4h equivalent)
    # Note: We'll use 4h as our primary timeframe for signal generation
    upper_donchian_aligned = align_htf_to_ltf(prices, df_12h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_12h, lower_donchian)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average (moderate to balance trades and filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian, above 1d EMA34, volume filter
        if (close[i] > upper_donchian_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian, below 1d EMA34, volume filter
        elif (close[i] < lower_donchian_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses the 1d EMA34 (trend reversal)
        elif position == 1 and close[i] < ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0