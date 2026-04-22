#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Load daily data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_1d_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, high_1d_roll)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, low_1d_roll)
    
    # 6h ATR for volatility filter (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_200_aligned[i]) or np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume surge AND weekly EMA200 uptrend
            if close[i] > donchian_high_1d[i] and vol_surge[i] and close[i] > ema_1w_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian low with volume surge AND weekly EMA200 downtrend
            elif close[i] < donchian_low_1d[i] and vol_surge[i] and close[i] < ema_1w_200_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian level
            if position == 1:
                if close[i] < donchian_low_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyEMA200_Trend_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0