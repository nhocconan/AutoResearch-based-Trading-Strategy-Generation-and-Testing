#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA50) and volume confirmation
# In bull/bear markets: breakout catches strong moves, EMA50 filter avoids counter-trend trades
# Volume confirmation ensures breakout validity. Discrete sizing 0.30 limits trades to ~10-25/year
# Works in ranging markets: filter reduces whipsaws by requiring alignment with weekly trend

name = "1d_1w_donchian_breakout_ema_volume_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period) based on prior period to avoid look-ahead
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian Low or weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian High or weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long on breakout above Donchian High with volume confirmation and bullish weekly trend
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Enter short on breakout below Donchian Low with volume confirmation and bearish weekly trend
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.30
    
    return signals