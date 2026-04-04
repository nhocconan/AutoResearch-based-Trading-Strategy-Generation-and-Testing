#!/usr/bin/env python3
# exp_6460_4h_donchian20_1d_ema_vol_v1
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Works in bull/bear: Donchian breakouts capture strong moves; EMA50 filters counter-trend noise; volume avoids false breakouts
# Target: 75-200 total trades over 4 years, Sharpe > 0 on all symbols
# Uses discrete position sizing (0.0, ±0.30) to minimize fee churn

name = "exp_6460_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if df_1d is None or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 (trend filter)
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d_50 = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Upper and lower bands
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 20 to ensure Donchian is valid
    for i in range(20, n):
        # Skip if EMA or volume data not ready
        if np.isnan(ema_1d_50_aligned[i]) or np.isnan(vol_ma[i]):
            continue
            
        # Long condition: price breaks above Donchian high + above 1d EMA50 + volume > 20 MA
        if (close[i] > donchian_high[i] and 
            close[i] > ema_1d_50_aligned[i] and 
            volume[i] > vol_ma[i]):
            if position != 1:
                signals[i] = 0.30  # Enter long
                position = 1
            else:
                signals[i] = 0.30  # Maintain long
        # Short condition: price breaks below Donchian low + below 1d EMA50 + volume > 20 MA
        elif (close[i] < donchian_low[i] and 
              close[i] < ema_1d_50_aligned[i] and 
              volume[i] > vol_ma[i]):
            if position != -1:
                signals[i] = -0.30  # Enter short
                position = -1
            else:
                signals[i] = -0.30  # Maintain short
        # Exit conditions: price retracement to midpoint or opposite Donchian touch
        elif position == 1:
            # Exit long if price retouches Donchian low or crosses below EMA50
            if close[i] <= donchian_low[i] or close[i] < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # Maintain long
        elif position == -1:
            # Exit short if price retouches Donchian high or crosses above EMA50
            if close[i] >= donchian_high[i] or close[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # Maintain short
        else:
            signals[i] = 0.0  # Flat
    
    return signals