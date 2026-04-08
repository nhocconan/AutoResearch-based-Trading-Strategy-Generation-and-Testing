# 4H DONCHIAN BREAKOUT WITH TREND FILTER AND VOLUME CONFIRMATION
# Hypothesis: Price breakout from Donchian Channel (20-period high/low) combined with 1-day trend filter (price > 200 EMA) and volume confirmation (>1.5x average volume) captures sustained trends in both bull and bear markets.
# The trend filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaw.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 20-50 trades per year on 4H timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for 200 EMA
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian Channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().values
    donchian_low = low_roll.min().values
    
    # Calculate 200 EMA on daily timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or trend changes
            if close[i] < donchian_low[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or trend changes
            if close[i] > donchian_high[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Donchian breakout entries with trend filter
            if close[i] > donchian_high[i] and volume_ok and close[i] > ema_200_1d_aligned[i]:
                # Long breakout above upper band with uptrend and volume
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and volume_ok and close[i] < ema_200_1d_aligned[i]:
                # Short breakdown below lower band with downtrend and volume
                position = -1
                signals[i] = -0.25
    
    return signals