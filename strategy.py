#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Weekly data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 20-period EMA on weekly close
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # 14-day ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, df_1d, atr_14d)
    
    # 20-day volume average
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current 12h price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14d_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA20 + volume surge + close > open (bullish candle)
            if (close[i] > ema_20_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma20_1d_aligned[i] and
                close[i] > prices['open'].values[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA20 + volume surge + close < open (bearish candle)
            elif (close[i] < ema_20_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma20_1d_aligned[i] and
                  close[i] < prices['open'].values[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above weekly EMA20
            if position == 1:
                if close[i] < ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA20_VolumeSurge_Direction_v1"
timeframe = "12h"
leverage = 1.0