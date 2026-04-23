#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
Williams %R identifies overbought/oversold conditions; trend filter ensures trades align with higher timeframe direction;
volume spike confirms institutional participation. Works in both bull/bear via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(highest_high_14[i]) or 
            np.isnan(lowest_low_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current volume > 2.0x 20-period MA (volume spike)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend AND volume spike
            if williams_r[i] < -80 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume spike
            elif williams_r[i] > -20 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -50) or reverse signal
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R rises above -50 (returns from oversold)
                if williams_r[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R falls below -50 (returns from overbought)
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0