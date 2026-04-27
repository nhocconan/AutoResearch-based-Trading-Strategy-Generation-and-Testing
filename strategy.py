#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike.
# Long when price breaks above 4h Donchian upper band with 1d uptrend and volume spike.
# Short when price breaks below 4h Donchian lower band with 1d downtrend and volume spike.
# Volume filter: current volume > 2x 20-period average.
# Exit: time-based exit after 3 bars or opposite signal.
# Designed for 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend and requiring volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_low)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Exit conditions: time-based exit after 3 bars or opposite signal
        if position == 1 and (bars_since_entry >= 3 or 
                              close[i] < donchian_low_aligned[i]):
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and (bars_since_entry >= 3 or 
                                 close[i] > donchian_high_aligned[i]):
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Entry conditions
            if position == 0:
                # Long: price breaks above Donchian high AND 1d uptrend AND volume spike
                if (close[i] > donchian_high_aligned[i] and 
                    close[i] > ema50_1d_aligned[i] and 
                    volume_filter[i]):
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Short: price breaks below Donchian low AND 1d downtrend AND volume spike
                elif (close[i] < donchian_low_aligned[i] and 
                      close[i] < ema50_1d_aligned[i] and 
                      volume_filter[i]):
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0