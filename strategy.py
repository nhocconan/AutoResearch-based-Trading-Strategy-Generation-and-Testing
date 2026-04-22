#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20-period) with weekly EMA200 trend filter and volume confirmation.
# Uses Donchian channel breakouts for trend following, with weekly EMA200 as primary trend filter
# and volume spikes (>1.5x 20-period average) for confirmation. Designed for low trade frequency
# (~10-20 trades/year) to minimize fee decay. Works in both bull and bear markets by following
# higher timeframe trend - only takes long signals in uptrend (price > weekly EMA200) and short
# signals in downtrend (price < weekly EMA200).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for EMA200 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 200-period EMA on weekly close for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily timeframe (waits for weekly bar to close)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Donchian channels (20-period high/low) on daily data
    high_max = prices['high'].rolling(window=20, min_periods=20).max().values
    low_min = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if np.isnan(ema_200_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        high = prices['high'].iloc[i]
        low = prices['low'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_200_aligned[i]
        donchian_high = high_max[i]
        donchian_low = low_min[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if high > donchian_high and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif low < donchian_low and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or trend breaks
                if low < donchian_low or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or trend breaks
                if high > donchian_high or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0