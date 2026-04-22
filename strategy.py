#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: Daily EMA34 trend + Weekly Pivot S1/R1 breakout with volume confirmation
    # Uses weekly pivot points for stronger support/resistance levels
    # Weekly trend filter avoids counter-trend trades
    # Volume surge confirms institutional participation
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA34 trend filter
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Weekly Pivot Points (S1, R1)
    # Using previous week's OHLC
    high_prev = np.roll(high_1w, 1)
    low_prev = np.roll(low_1w, 1)
    close_prev = np.roll(close_1w, 1)
    # First value will be NaN due to roll, handled by isnan check
    pp = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pp - low_prev
    s1 = 2 * pp - high_prev
    
    # Align weekly levels to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Daily ATR for volatility filter
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
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S1 with volume surge AND weekly EMA34 uptrend
            if close[i] > s1_aligned[i] and vol_surge[i] and close[i] > ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume surge AND weekly EMA34 downtrend
            elif close[i] < r1_aligned[i] and vol_surge[i] and close[i] < ema_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to EMA34 level (dynamic stop)
            if position == 1:
                if close[i] < ema_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_S1_R1_Breakout_1wEMA34_Trend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0