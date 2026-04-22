#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for ATR-based range
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only long in uptrend (price > weekly EMA50), only short in downtrend
        is_uptrend = prices['close'].iloc[i] > ema_50_1w_aligned[i]
        is_downtrend = prices['close'].iloc[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Pullback to weekly EMA50 + 0.5*ATR with volume surge in uptrend
            if is_uptrend and prices['low'].iloc[i] <= (ema_50_1w_aligned[i] + 0.5 * atr_1d_aligned[i]) and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Pullback to weekly EMA50 - 0.5*ATR with volume surge in downtrend
            elif is_downtrend and prices['high'].iloc[i] >= (ema_50_1w_aligned[i] - 0.5 * atr_1d_aligned[i]) and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly EMA50
            if position == 1:
                if prices['close'].iloc[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if prices['close'].iloc[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_EMA50_Pullback_1wTrend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0