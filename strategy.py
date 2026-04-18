#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour donchian breakout with 12-hour EMA trend filter and volume confirmation.
# Donchian(20) breakout provides clear entry/exit signals.
# 12h EMA34 filters for trend direction to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h data
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA34 on 12h data
    close_12h = pd.Series(df_12h['close'].values)
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian band AND price > 12h EMA34 AND volume spike
            if close[i] > upper[i] and close[i] > ema_34_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band AND price < 12h EMA34 AND volume spike
            elif close[i] < lower[i] and close[i] < ema_34_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below lower Donchian band OR volume dries up
            if close[i] < lower[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above upper Donchian band OR volume dries up
            if close[i] > upper[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals