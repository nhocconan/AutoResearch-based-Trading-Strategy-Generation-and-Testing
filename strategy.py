# 4h_1d_donchian_breakout_volume_v1
# Strategy: 4h Donchian breakout with volume confirmation and ATR stoploss
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian(20) breakouts capture trend continuation. Volume > 1.5x 20-period average confirms institutional participation.
# Works in bull markets via upward breakouts and bear markets via downward breakouts. ATR-based stoploss manages risk.
# Designed for low trade frequency (~20-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for stoploss and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(atr[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max[i-1]  # Break above previous high
        breakout_down = close[i] < low_min[i-1]  # Break below previous low
        
        # Entry conditions
        # Long: Upward breakout AND volume confirmation
        if breakout_up and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Downward breakout AND volume confirmation
        elif breakout_down and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: ATR-based stoploss
        elif position == 1 and close[i] < high_max[i-1] - 2.0 * atr[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > low_min[i-1] + 2.0 * atr[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals