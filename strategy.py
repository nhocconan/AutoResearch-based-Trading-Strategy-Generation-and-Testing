#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h 4h/1d Confluence with Volume and Session Filter
# Hypothesis: Combines 4h trend (EMA21), 1d momentum (ROC20), and volume confirmation on 1h timeframe.
# Trades only during active session (08-20 UTC) to avoid low-volume noise.
# Uses 4h EMA21 for trend direction, 1d ROC20 for momentum filter, and 1h volume spike for entry timing.
# Target: 15-35 trades/year (60-140 over 4 years) to minimize fee drag.
name = "1h_4h_1d_confluence_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ROC(20) for momentum
    close_1d = df_1d['close'].values
    roc_1d = np.full_like(close_1d, np.nan)
    roc_1d[20:] = (close_1d[20:] - close_1d[:-20]) / close_1d[:-20] * 100
    roc_1d_1h = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (already datetime64[ms] index)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_1h[i]) or np.isnan(roc_1d_1h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Require session and volume filter
        if not (session_filter[i] and vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # Long: 4h uptrend + positive 1d momentum
        if close[i] > ema_4h_1h[i] and roc_1d_1h[i] > 0:
            signals[i] = 0.20
        # Short: 4h downtrend + negative 1d momentum
        elif close[i] < ema_4h_1h[i] and roc_1d_1h[i] < 0:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals