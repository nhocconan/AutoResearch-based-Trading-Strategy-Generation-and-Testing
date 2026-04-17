#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > 1w EMA200 AND volume > 1.5x 20-period average volume.
Short when price breaks below Donchian(20) low AND price < 1w EMA200 AND volume > 1.5x 20-period average volume.
Exit when price touches Donchian(20) midpoint or trend reverses.
Uses 1d for price action and Donchian channels, 1w for higher-timeframe trend filter.
Target: 30-100 total trades over 4 years (7-25/year). Donchian breakouts capture strong momentum,
weekly EMA200 filters for higher-timeframe trend alignment, volume confirmation reduces false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Donchian(20) on 1d timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.5x 20-period average volume
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma20)
    
    # Align 1w EMA200 to 1d timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_confirm = volume_confirm[i]
        ema200 = ema200_1w_aligned[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_mid[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > 1w EMA200 AND volume confirmation
            if price > d_high and price > ema200 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1w EMA200 AND volume confirmation
            elif price < d_low and price < ema200 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches Donchian midpoint OR price < 1w EMA200 (trend reversal)
            if price <= d_mid or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches Donchian midpoint OR price > 1w EMA200 (trend reversal)
            if price >= d_mid or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA200_VolumeConfirm"
timeframe = "1d"
leverage = 1.0