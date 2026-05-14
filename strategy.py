#!/usr/bin/env python3
# Hypothesis: 12h Bollinger Band squeeze breakout with 1d trend filter (EMA50) and volume confirmation (>1.5x 20-period average).
# Long when price breaks above upper BB AND close > 1d EMA50 AND volume > 1.5x MA20.
# Short when price breaks below lower BB AND close < 1d EMA50 AND volume > 1.5x MA20.
# Exit when price crosses BB middle (20 SMA) OR trend filter fails.
# Bollinger Band squeeze identifies low volatility periods preceding breakouts.
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe to minimize fee drag.

name = "12h_BollingerSqueeze_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # --- 12h Indicators (LTF) ---
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # 20 SMA for exit
    
    # 12h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after BB warmup
        # Skip if missing data
        if (np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper BB AND close > 1d EMA50 (bullish trend) AND volume confirm
            if (close[i] > upper_bb[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower BB AND close < 1d EMA50 (bearish trend) AND volume confirm
            elif (close[i] < lower_bb[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below middle BB OR trend fails (close < 1d EMA50)
            if (close[i] < middle_bb[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above middle BB OR trend fails (close > 1d EMA50)
            if (close[i] > middle_bb[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals