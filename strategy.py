#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w EMA filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1w EMA(50) AND volume > 1.5x average
# Exit when price returns to Donchian midpoint or volume dries up
# Uses 12h timeframe to reduce trade frequency (target: 75-150 total trades over 4 years)
# Works in both bull/bear markets by following trend (via 1w EMA) and using volume confirmation

name = "12h_donchian_1w_ema_vol_v1"
timeframe = "12h"
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
    
    # Donchian Channel (20-period) on 12h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # 1-week EMA (50-period) for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_1w = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to Donchian midpoint OR volume drops below threshold
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend filter, and volume confirmation
            # Long: price breaks above Donchian high AND above 1w EMA AND volume confirmation
            if (close[i] > donch_high[i] and close[i] > ema_1w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1w EMA AND volume confirmation
            elif (close[i] < donch_low[i] and close[i] < ema_1w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals