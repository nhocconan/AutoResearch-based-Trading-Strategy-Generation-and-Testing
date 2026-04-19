#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour Donchian breakout with 1-day volume filter and session filter (08-20 UTC).
# Uses 4h Donchian(20) for trend direction, 1d volume spike for confirmation, 1h for entry timing.
# Designed for 1h timeframe to capture medium-term breakouts with low frequency (~15-25 trades/year).
# Entry: Long when 1h close > 4h Donchian upper band and volume > 1.5x 20-day avg and session active.
# Short when 1h close < 4h Donchian lower band and volume > 1.5x 20-day avg and session active.
# Exit: Opposite Donchian band touch or volume drop below average.
# Uses strict conditions to limit trades and avoid overtrading.

name = "1h_Donchian20_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian bands on 4h data
    donchian_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (waits for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1-day volume spike filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_20d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (volume_20d * 1.5)
    
    # Align volume spike to 1h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC (precompute for efficiency)
    session_hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or not in session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 4h Donchian high with volume confirmation
            if (close[i] > donchian_high_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below 4h Donchian low with volume confirmation
            elif (close[i] < donchian_low_aligned[i] and volume_spike_aligned[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price touches 4h Donchian low or volume drops
            if (close[i] < donchian_low_aligned[i]) or (not volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price touches 4h Donchian high or volume drops
            if (close[i] > donchian_high_aligned[i]) or (not volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals