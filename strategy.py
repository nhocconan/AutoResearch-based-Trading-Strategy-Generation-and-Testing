#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action with 4h Donchian breakout and volume confirmation.
# Long when: 4h Donchian upper breaks, volume > 2x 20-period avg, and price > 1h VWAP
# Short when: 4h Donchian lower breaks, volume > 2x 20-period avg, and price < 1h VWAP
# Uses 4h for direction (reduces false signals), 1h for entry timing.
# Target: 15-30 trades/year per symbol.
name = "1h_DonchianBreakout_Volume_VWAP"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, 0)
    
    # 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: 4h Donchian upper break, volume spike, price > VWAP
            if (price > donchian_high_1h[i] and 
                vol > 2.0 * vol_ma and 
                price > vwap[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: 4h Donchian lower break, volume spike, price < VWAP
            elif (price < donchian_low_1h[i] and 
                  vol > 2.0 * vol_ma and 
                  price < vwap[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price < 4h Donchian lower or volume drop
            if (price < donchian_low_1h[i] or vol < 0.5 * vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price > 4h Donchian upper or volume drop
            if (price > donchian_high_1h[i] or vol < 0.5 * vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals