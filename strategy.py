#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
Williams %R identifies overbought/oversold conditions; EMA34 filters for trend alignment to avoid counter-trend trades;
volume spike confirms institutional participation. Works in both bull/bear via trend filter and mean reversion logic.
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
    
    # Calculate 12h Williams %R (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low + 1e-10) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 12h volume > 2.0x 20-period MA (stricter to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            if williams_r_aligned[i] < -80 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            elif williams_r_aligned[i] > -20 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses -50 (mean reversion midpoint) or opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long on Williams %R > -50 (reversion from oversold)
                if williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short on Williams %R < -50 (reversion from overbought)
                if williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0