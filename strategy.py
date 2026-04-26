#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
- Long when price breaks above upper Donchian(20) with volume spike and 12h uptrend
- Short when price breaks below lower Donchian(20) with volume spike and 12h downtrend
- Volume spike confirms institutional participation (2x 20-period average)
- 12h trend filter reduces whipsaw in bear markets (2022) and captures major moves
- Designed for low trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    # Calculate Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA50, 20 for Donchian and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and trend filter
        price_above_upper = close[i] > upper_channel[i]
        price_below_lower = close[i] < lower_channel[i]
        
        # 12h trend filter
        trend_up = close[i] > ema50_12h_aligned[i]
        trend_down = close[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume spike AND 12h uptrend
            if price_above_upper and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND volume spike AND 12h downtrend
            elif price_below_lower and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below lower Donchian OR 12h trend turns down
            if price_below_lower or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above upper Donchian OR 12h trend turns up
            if price_above_upper or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0