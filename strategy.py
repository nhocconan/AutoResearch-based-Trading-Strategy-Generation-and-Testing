#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d VWAP trend filter and volume confirmation.
# Long when: Price breaks above 12h Donchian high AND price > 1d VWAP AND volume > 2x 20-period average
# Short when: Price breaks below 12h Donchian low AND price < 1d VWAP AND volume > 2x 20-period average
# Exit when: Price crosses back below/above 12h Donchian mid-point
# Uses 1d VWAP for trend filter (works in bull/bear via mean-reversion to value),
# Donchian for breakout signals, volume for confirmation.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).
name = "12h_Donchian20_1dVWAP_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d VWAP ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # VWAP = cumulative (typical price * volume) / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # 12h Donchian channels
    df_12h = get_htf_data(prices, '12h')
    donch_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian high AND price > 1d VWAP AND volume spike
            if (price > donch_high_aligned[i] and 
                price > vwap_1d_aligned[i] and 
                vol > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low AND price < 1d VWAP AND volume spike
            elif (price < donch_low_aligned[i] and 
                  price < vwap_1d_aligned[i] and 
                  vol > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian mid-point
            if price < donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Donchian mid-point
            if price > donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals