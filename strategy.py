#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses Donchian channels for breakout signals, 12h EMA34 for trend direction,
# and volume spike for confirmation. Designed for low-frequency, high-conviction
# trades that work in both bull and bear markets by filtering with trend.
# Entry: Long when price breaks above Donchian(20) upper band AND 12h EMA34 up AND volume spike.
#        Short when price breaks below Donchian(20) lower band AND 12h EMA34 down AND volume spike.
# Exit: Opposite Donchian band touch or trend reversal.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_Donchian20_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with uptrend and volume
            if (close[i] > donchian_upper[i] and 
                ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with downtrend and volume
            elif (close[i] < donchian_lower[i] and 
                  ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower band or trend turns down
            if (close[i] < donchian_lower[i]) or (ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper band or trend turns up
            if (close[i] > donchian_upper[i]) or (ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals