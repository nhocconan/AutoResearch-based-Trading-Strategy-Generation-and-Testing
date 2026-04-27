#!/usr/bin/env python3
"""
1d_Weekly_Trend_Follower_v3
Hypothesis: Use 1-week EMA10 trend direction on daily timeframe with volume confirmation (1.5x 20-day average) and ATR(14) volatility filter (>0.5% of price). Long when price > weekly EMA10, short when price < weekly EMA10. Exit on opposite signal. Designed for low turnover (target 10-20 trades/year) to minimize fee decay while capturing major trends in both bull and bear markets.
"""

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
    
    # Calculate weekly EMA10 from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    ema_10 = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # ATR(14) volatility filter: only trade when ATR > 0.5% of price
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    vol_filter = atr > (0.005 * close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, volume average, and ATR
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_10_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        ema_val = ema_10_aligned[i]
        vol_conf = volume_confirm[i]
        vol_filt = vol_filter[i]
        
        if position == 0:
            # Long: price above weekly EMA10 with volume and volatility confirmation
            if close[i] > ema_val and vol_conf and vol_filt:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA10 with volume and volatility confirmation
            elif close[i] < ema_val and vol_conf and vol_filt:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA10
            if close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA10
            if close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Trend_Follower_v3"
timeframe = "1d"
leverage = 1.0