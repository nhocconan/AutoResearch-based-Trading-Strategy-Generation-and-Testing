#!/usr/bin/env python3
"""
1d_weekly_trend_ema_volume_v1
Hypothesis: Weekly EMA trend with daily volume confirmation and ATR volatility filter.
Long when daily price closes above weekly EMA50 with above-average volume.
Short when daily price closes below weekly EMA50 with above-average volume.
Uses ATR to filter out low-volatility periods where breakouts fail.
Designed to work in both bull (trend following) and bear (mean reversion at extremes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_trend_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly close
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # Align EMA50 to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(close[i]) or np.isnan(volume[i])):
            continue
            
        # Volatility filter: only trade when ATR > 50% of its 50-period average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma[i]) or atr[i] < 0.5 * atr_ma[i]:
            signals[i] = 0.0
            continue
            
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Long: price above weekly EMA50 with volume confirmation
        if close[i] > ema_50_aligned[i] and vol_confirmed:
            signals[i] = 0.25
        # Short: price below weekly EMA50 with volume confirmation
        elif close[i] < ema_50_aligned[i] and vol_confirmed:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals