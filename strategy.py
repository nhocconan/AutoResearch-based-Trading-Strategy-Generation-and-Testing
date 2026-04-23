#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume confirmation.
Target: 15-30 trades/year per symbol. Uses 4h for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise trades. Discrete position sizing (0.20) minimizes fee churn.
Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter (loaded ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h Camarilla levels (using previous 1h bar's range)
    # Note: we need previous bar's high/low, so we shift by 1
    camarilla_r1 = close + (high - low) * 1.1 / 12  # R1 = C + (H-L)*1.1/12
    camarilla_s1 = close - (high - low) * 1.1 / 12  # S1 = C - (H-L)*1.1/12
    # Shift to use previous bar's levels (no look-ahead)
    camarilla_r1 = np.roll(camarilla_r1, 1)
    camarilla_s1 = np.roll(camarilla_s1, 1)
    camarilla_r1[0] = np.nan
    camarilla_s1[0] = np.nan
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # need EMA34, volume MA20, and shifted Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 4h EMA34 = uptrend, close < 4h EMA34 = downtrend
        trend_up = close[i] > ema_34_4h_aligned[i]
        trend_down = close[i] < ema_34_4h_aligned[i]
        
        # Volume filter: 1h volume > 1.5x 20-period MA (balanced to reduce trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend AND volume confirmation
            if close[i] > camarilla_r1[i] and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s1[i] and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: break of opposite Camarilla level
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S1
                if close[i] < camarilla_s1[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R1
                if close[i] > camarilla_r1[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeConfirmation_Session"
timeframe = "1h"
leverage = 1.0