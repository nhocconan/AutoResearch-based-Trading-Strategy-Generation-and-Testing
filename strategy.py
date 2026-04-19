#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h volume confirmation and 1d trend filter.
# Uses 4h volume spike to confirm institutional interest and 1d EMA200 for trend direction.
# Designed to work in both bull and bear markets by filtering breakouts with volume and trend.
# Target: 15-37 trades/year per symbol.
name = "1h_Donchian20_VolumeTrend_EMA200"
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
    
    # Get 4h data for volume confirmation (spike detection)
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian Channel (20-period) on 1h
    donch_len = 20
    upper_dc = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_dc = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Calculate 4h volume average (20-period) for spike detection
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h volume MA and 1d EMA200 to 1h timeframe
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, 20, 200)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_dc[i]
        lower = lower_dc[i]
        vol_ma = vol_ma_4h_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 2.0x 4h average volume
        volume_confirmed = vol > 2.0 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper
        bearish_breakout = price < lower
        
        if position == 0:
            # Look for entry: breakout in direction of 1d trend with volume confirmation
            if bullish_breakout and (price > ema_200) and volume_confirmed:
                signals[i] = 0.20
                position = 1
            elif bearish_breakout and (price < ema_200) and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price returns to 50% of the Donchian channel or trend changes
            mid_point = (upper + lower) * 0.5
            if price < mid_point:  # Return to midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price returns to 50% of the Donchian channel
            mid_point = (upper + lower) * 0.5
            if price > mid_point:  # Return to midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals