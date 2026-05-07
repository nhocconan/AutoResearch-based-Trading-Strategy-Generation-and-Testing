#!/usr/bin/env python3
name = "12h_1dSupertrend_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily Supertrend (10, 3.0)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR calculation
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend upper and lower bands
    hl2_1d = (high_1d + low_1d) / 2
    upper_band_1d = hl2_1d + 3.0 * atr_1d
    lower_band_1d = hl2_1d - 3.0 * atr_1d
    
    # Initialize Supertrend
    supertrend_1d = np.full(len(close_1d), np.nan)
    direction_1d = np.full(len(close_1d), 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close_1d)):
        if np.isnan(atr_1d[i]) or np.isnan(upper_band_1d[i]) or np.isnan(lower_band_1d[i]):
            continue
        
        if i == 10:
            supertrend_1d[i] = lower_band_1d[i]
            direction_1d[i] = 1
        else:
            if close_1d[i] > upper_band_1d[i-1]:
                direction_1d[i] = 1
            elif close_1d[i] < lower_band_1d[i-1]:
                direction_1d[i] = -1
            else:
                direction_1d[i] = direction_1d[i-1]
                if direction_1d[i] == 1 and lower_band_1d[i] < supertrend_1d[i-1]:
                    lower_band_1d[i] = supertrend_1d[i-1]
                if direction_1d[i] == -1 and upper_band_1d[i] > supertrend_1d[i-1]:
                    upper_band_1d[i] = supertrend_1d[i-1]
            
            if direction_1d[i] == 1:
                supertrend_1d[i] = lower_band_1d[i]
            else:
                supertrend_1d[i] = upper_band_1d[i]
    
    # Align daily Supertrend to 12h
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    # Calculate weekly trend (EMA 34)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period on 12h)
    high_low = high - low
    high_close = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34)  # Wait for volume MA, ATR, and weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(supertrend_1d_aligned[i]) or np.isnan(direction_1d_aligned[i]) or \
           np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Supertrend (bullish), above weekly EMA34, volume spike
            if (close[i] > supertrend_1d_aligned[i] and 
                direction_1d_aligned[i] == 1 and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price < Supertrend (bearish), below weekly EMA34, volume spike
            elif (close[i] < supertrend_1d_aligned[i] and 
                  direction_1d_aligned[i] == -1 and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < Supertrend or below weekly EMA34
            if (close[i] < supertrend_1d_aligned[i] or 
                direction_1d_aligned[i] == -1 or
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > Supertrend or above weekly EMA34
            if (close[i] > supertrend_1d_aligned[i] or 
                direction_1d_aligned[i] == 1 or
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Supertrend with weekly trend filter and volume confirmation.
# Uses daily Supertrend (10, 3.0) for trend direction and entry signals.
# Weekly EMA(34) ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy when daily Supertrend flips bullish in weekly uptrend) 
# and bear markets (sell when daily Supertrend flips bearish in weekly downtrend).
# Position size 0.25 balances risk and keeps trade frequency moderate (~15-25 trades/year).