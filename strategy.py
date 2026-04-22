#!/usr/bin/env python3
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
    
    # Load weekly data for trend filter and ATR (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_40_1w = close_1w_series.ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Weekly ATR(20) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # first bar
    tr2[0] = high_1w[0] - close_1w_arr[0]
    tr3[0] = low_1w[0] - close_1w_arr[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20_1w = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_20_1w)
    
    # Load daily data for price levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(atr_20_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume filter AND weekly uptrend
            if (close[i] > high_20_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i] and
                close[i] > ema_40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume filter AND weekly downtrend
            elif (close[i] < low_20_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i] and
                  close[i] < ema_40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses midline or opposite band
            if position == 1:
                # Exit long: Price closes below 20-day low
                if close[i] < low_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above 20-day high
                if close[i] > high_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_Donchian20_WeeklyEMA40_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0