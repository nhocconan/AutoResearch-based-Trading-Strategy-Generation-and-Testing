#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: Daily EMA34 trend + Camarilla S1/R1 breakout with volume confirmation
    # Uses tighter breakout levels (S1/R1) for more reliable signals, reducing false breakouts
    # Works in bull markets (buy S1 breakout) and bear markets (sell R1 breakdown)
    # Volume surge filters low-probability breakouts
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA34 trend filter
    ema_1d_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Daily Camarilla pivot levels (S1, R1 only - tighter levels)
    range_1d = high_1d - low_1d
    close_prev = close_1d
    s1_1d = close_prev - (range_1d * 1.0 / 6)
    r1_1d = close_prev + (range_1d * 1.0 / 6)
    
    # Align daily levels to 12h
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # 12h ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(s1_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(ema_1d_34_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S1 with volume surge AND daily EMA34 uptrend
            if close[i] > s1_1d_aligned[i] and vol_surge[i] and close[i] > ema_1d_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume surge AND daily EMA34 downtrend
            elif close[i] < r1_1d_aligned[i] and vol_surge[i] and close[i] < ema_1d_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to EMA34 level (dynamic stop)
            if position == 1:
                if close[i] < ema_1d_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_1d_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_S1_R1_Breakout_1dEMA34_Trend_VolumeSurge_v1"
timeframe = "12h"
leverage = 1.0