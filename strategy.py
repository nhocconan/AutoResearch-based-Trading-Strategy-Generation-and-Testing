#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d VWAP filter and volume confirmation
# Enter long when price breaks above Donchian(20) high with volume > 1.3x average and price > 1d VWAP
# Enter short when price breaks below Donchian(20) low with volume > 1.3x average and price < 1d VWAP
# Uses volume filter and VWAP trend filter to limit trades to 75-200 total over 4 years
# Exit when price crosses Donchian middle or reverses against VWAP trend

name = "6h_donchian_1d_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 6h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d VWAP for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    cum_vol = np.cumsum(volume_1d)
    cum_pv = np.cumsum(typical_price_1d * volume_1d)
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian middle OR price < 1d VWAP
            if close[i] < donchian_mid[i] or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian middle OR price > 1d VWAP
            if close[i] > donchian_mid[i] or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and VWAP filter
            if close[i] > donchian_high[i] and volume[i] > volume_threshold[i] and close[i] > vwap_1d_aligned[i]:
                # Long breakout in uptrend (above VWAP)
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i] and volume[i] > volume_threshold[i] and close[i] < vwap_1d_aligned[i]:
                # Short breakdown in downtrend (below VWAP)
                signals[i] = -0.25
                position = -1
    
    return signals