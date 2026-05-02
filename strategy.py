#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels identify key support/resistance levels where price often breaks out with momentum
# Weekly EMA50 provides long-term trend filter to align with dominant momentum and reduce counter-trend whipsaws
# Volume spike (2.0x 20-period average) confirms breakout conviction
# Targets 30-100 trades over 4 years (7-25/year) for 1d timeframe
# Works in both bull and bear markets by capturing breakouts in direction of higher timeframe trend

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels (20-period) from previous completed 1d bar
    # We use rolling window on the primary timeframe data but shift to avoid look-ahead
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for rolling calculations and HTF data alignment)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper + price > 1w EMA50 + volume spike
            if close[i] > high_roll[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + price < 1w EMA50 + volume spike
            elif close[i] < low_roll[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower (reversal signal)
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper (reversal signal)
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals