#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and session filter (08-20 UTC).
# Long when price breaks above 4h Donchian high (20) with volume > 1.5x 24-period average and in session.
# Short when price breaks below 4h Donchian low (20) with same conditions.
# Exit when price crosses back over 4h Donchian midpoint.
# Uses 4h Donchian for trend direction, volume for conviction, session to reduce noise.
# Designed for ~15-30 trades/year per symbol (60-120 over 4 years).
name = "1h_4hDonchian20_Volume_Session_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    
    # Donchian(20) on 4h high/low
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid_4h = (donch_high_4h + donch_low_4h) / 2.0
    
    # Align to 1h timeframe
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    donch_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 1h = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(donch_mid_4h_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high = donch_high_4h_aligned[i]
        donch_low = donch_low_4h_aligned[i]
        donch_mid = donch_mid_4h_aligned[i]
        vol_filter = volume_filter[i]
        sess_filter = session_filter[i]
        
        if position == 0:
            # Long: price above Donchian high with volume and session
            if close_val > donch_high and vol_filter and sess_filter:
                signals[i] = 0.20
                position = 1
            # Short: price below Donchian low with volume and session
            elif close_val < donch_low and vol_filter and sess_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below Donchian midpoint
            if close_val < donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above Donchian midpoint
            if close_val > donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals