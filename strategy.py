#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h trend filter using Donchian breakout and volume confirmation.
# Uses 4h Donchian channel (20) for trend direction, 1h volume spike for entry timing,
# and session filter (08-20 UTC) to reduce noise. Designed to capture breakouts in both
# bull and bear markets with tight entry conditions to limit trades (target: 15-37/year).
# Exit on mean reversion to 4h Donchian middle or volatility contraction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for Donchian channel (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian Channel (20-period)
    donch_len = 20
    upper = pd.Series(high_4h).rolling(window=donch_len, min_periods=donch_len).max().values
    lower = pd.Series(low_4h).rolling(window=donch_len, min_periods=donch_len).min().values
    middle = (upper + lower) / 2
    
    # Align Donchian to 1h timeframe (wait for 4h close)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle)
    
    # Calculate 20-period average volume for volume spike detection (1h)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        middle_val = middle_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (tight filter)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian upper with volume spike
            if price > upper_band and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian lower with volume spike
            elif price < lower_band and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on retracement to 4h Donchian middle (mean reversion)
                if price < middle_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on retracement to 4h Donchian middle (mean reversion)
                if price > middle_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0