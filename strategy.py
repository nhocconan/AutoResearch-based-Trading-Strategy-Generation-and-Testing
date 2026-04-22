#!/usr/bin/env python3
"""
1H 4H-1D Trend Reversion with Volume Confirmation
Hypothesis: Price tends to revert to the 4H EMA20 after moving 1.5*ATR away, 
but only in the direction of the 1D trend (EMA50). Volume confirms momentum.
Works in both bull/bear: In uptrends, buy dips to EMA20; in downtrends, sell rallies to EMA20.
Targets 15-30 trades/year by requiring 4H/1D alignment + volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1H ATR for entry distance
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = tr2[0] if not np.isnan(tr2[0]) else 0
    tr3[0] = tr3[0] if not np.isnan(tr3[0]) else 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 4H data for EMA20 (dynamic mean reversion target)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h_20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_20_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_20)
    
    # Load 1D data for EMA50 (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_20_aligned[i]) or np.isnan(ema_1d_50_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1D uptrend + price near 4H EMA20 (within 0.5*ATR) + volume confirmation
            if (close[i] > ema_1d_50_aligned[i] and  # 1D uptrend
                close[i] >= ema_4h_20_aligned[i] - 0.5 * atr[i] and  # near 4H EMA20
                close[i] <= ema_4h_20_aligned[i] + 0.5 * atr[i] and
                volume[i] > 1.5 * vol_ma[i]):  # volume spike
                signals[i] = 0.20
                position = 1
            # Short: 1D downtrend + price near 4H EMA20 + volume confirmation
            elif (close[i] < ema_1d_50_aligned[i] and  # 1D downtrend
                  close[i] >= ema_4h_20_aligned[i] - 0.5 * atr[i] and  # near 4H EMA20
                  close[i] <= ema_4h_20_aligned[i] + 0.5 * atr[i] and
                  volume[i] > 1.5 * vol_ma[i]):  # volume spike
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price moves 1.5*ATR away from 4H EMA20 (mean reversion failed) or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: price drops below 4H EMA20 - 1.5*ATR or 1D turns down
                if (close[i] < ema_4h_20_aligned[i] - 1.5 * atr[i] or
                    close[i] < ema_1d_50_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above 4H EMA20 + 1.5*ATR or 1D turns up
                if (close[i] > ema_4h_20_aligned[i] + 1.5 * atr[i] or
                    close[i] > ema_1d_50_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4H_EMA20_1D_EMA50_MeanReversion_Volume"
timeframe = "1h"
leverage = 1.0