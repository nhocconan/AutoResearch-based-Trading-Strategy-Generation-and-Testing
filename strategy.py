#/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend
Breakout strategy using 4h Donchian channels with volume confirmation and 12h trend filter.
Long: Price breaks above Donchian(20) high + volume > 1.5x average + 12h EMA34 > EMA89
Short: Price breaks below Donchian(20) low + volume > 1.5x average + 12h EMA34 < EMA89
Exit: Opposite breakout or trend reversal
Designed for ~20-50 trades/year per symbol (80-200 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel parameters
    donchian_period = 20
    
    # Calculate Donchian channels
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_89_12h)
    
    # Volume confirmation - 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 89)  # need enough for Donchian and EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(ema_89_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_12h_aligned[i] > ema_89_12h_aligned[i]
        downtrend = ema_34_12h_aligned[i] < ema_89_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakdown_down = close[i] < donchian_low[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above Donchian high
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below Donchian low
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below Donchian low
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above Donchian high
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0