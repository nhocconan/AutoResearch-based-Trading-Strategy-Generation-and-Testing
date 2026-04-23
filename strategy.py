#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams %R extremes with 1h EMA trend filter and volume confirmation.
Long when 1d Williams %R < -80 (oversold) AND price > 1h EMA34 AND volume > 1.3x 20-period average.
Short when 1d Williams %R > -20 (overbought) AND price < 1h EMA34 AND volume > 1.3x 20-period average.
Exit when price crosses 1h EMA34 or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Williams %R identifies overextended moves; fading extremes with trend/volume filters captures mean reversion in ranging markets and pullbacks in trends.
Designed for 4h timeframe to target 20-50 trades/year per symbol (80-200 total over 4 years).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1h EMA34 for trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 34:
        return np.zeros(n)
    
    ema_34_1h = pd.Series(df_1h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_34_1h)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, vol MA needs 20, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr_val = williams_r_aligned[i]
        ema_val = ema_34_1h_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1h EMA34 AND volume spike
            if (wr_val < -80 and price > ema_val and volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Williams %R overbought (> -20) AND price < 1h EMA34 AND volume spike
            elif (wr_val > -20 and price < ema_val and volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price crosses 1h EMA34 (trend change)
            if position == 1 and price <= ema_val:
                exit_signal = True
            elif position == -1 and price >= ema_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Extremes_1hEMA34_TrendFilter_VolumeConfirmation_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0