#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA34 trend filter
# Breakouts above/below Donchian channels capture trending moves.
# Volume confirmation ensures institutional participation.
# 12h EMA34 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Works in bull markets (breakouts above upper channel) and bear markets (breakdowns below lower channel).
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels (20-period) on 4h data
    # Use rolling window with min_periods to avoid look-ahead
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Calculate EMA34 on 12h data for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        ema_val = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Close above upper channel AND price above EMA34 AND volume spike
            if close_val > upper_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close below lower channel AND price below EMA34 AND volume spike
            elif close_val < lower_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA34 (trend change) or at lower channel (stop loss)
            if close_val < ema_val or close_val <= lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: Close above EMA34 (trend change) or at upper channel (stop loss)
            if close_val > ema_val or close_val >= upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals