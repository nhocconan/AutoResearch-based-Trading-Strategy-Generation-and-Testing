#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume spike for 1h timeframe.
# Long when price breaks above 4h Donchian high with 4h uptrend and volume spike.
# Short when price breaks below 4h Donchian low with 4h downtrend and volume spike.
# Uses 4h for signal direction, 1h only for entry timing to reduce trade frequency.
# Volume filter: current volume > 2x 20-period average. Designed for 15-35 trades/year per symbol.
# Works in both bull and bear markets by following the 4h trend and requiring volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 20-period EMA on 4h close for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Session filter: 8-20 UTC (08:00 to 20:00)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above 4h Donchian high AND 4h uptrend AND volume spike
        if (close[i] > donchian_high_aligned[i] and 
            close[i] > ema20_4h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short conditions: price breaks below 4h Donchian low AND 4h downtrend AND volume spike
        elif (close[i] < donchian_low_aligned[i] and 
              close[i] < ema20_4h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_DonchianBreakout_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0