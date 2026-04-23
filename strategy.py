#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
Target: 25-50 trades/year per symbol (100-200 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
Williams %R identifies overbought/oversold conditions; EMA34 filter ensures trades align with higher timeframe trend; volume confirmation avoids false signals.
Works in both bull/bear via 1d trend filter and mean reversion logic.
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
    
    # Calculate Williams %R (14-period) on 4h data
    def calculate_williams_r(high, low, close, window=14):
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.values
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # need EMA34, volume MA20, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA (balanced to avoid overtrading)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend AND volume confirmation
            if williams_r[i] < -80 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume confirmation
            elif williams_r[i] > -20 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -50) or reverse signal
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R rises above -50 (mean reversion)
                if williams_r[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R falls below -50 (mean reversion)
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0