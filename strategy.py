#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Donchian channels capture breakout momentum, EMA50 filters for higher timeframe trend alignment.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Designed for low trade frequency (~15-25/year) to minimize fee decay.
# Works in bull markets via breakouts and bear markets via trend-following shorts.
# Daily timeframe avoids excessive churn while capturing multi-day moves.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d data
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (identity for same timeframe, but required for interface)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Use daily price and volume directly (no alignment needed for same timeframe)
    price = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        p = price[i]
        v = volume[i]
        dh = donch_high_aligned[i]
        dl = donch_low_aligned[i]
        ema = ema_50_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = v > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: break above Donchian high + uptrend + volume spike
            if p > dh and p > ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + downtrend + volume spike
            elif p < dl and p < ema and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or trend breaks
                if p < dl or p < ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or trend breaks
                if p > dh or p > ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0