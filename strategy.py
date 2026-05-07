#!/usr/bin/env python3
name = "1h_4h1d_Confluence_RSI_Vol"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h RSI(14) for trend filter
    delta_4h = pd.Series(df_4h['close']).diff()
    gain_4h = (delta_4h.where(delta_4h > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss_4h = (-delta_4h.where(delta_4h < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs_4h = gain_4h / loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.fillna(50).values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d EMA(50) for trend direction
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h volume spike: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for vol MA and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: 4h RSI > 55 (bullish), price > 1d EMA50, volume spike, in session
            if (rsi_4h_aligned[i] > 55 and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i] and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: 4h RSI < 45 (bearish), price < 1d EMA50, volume spike, in session
            elif (rsi_4h_aligned[i] < 45 and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i] and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 4h RSI < 40 or price < 1d EMA50
            if (rsi_4h_aligned[i] < 40 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 4h RSI > 60 or price > 1d EMA50
            if (rsi_4h_aligned[i] > 60 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h strategy using 4h RSI for momentum and 1d EMA50 for trend filter.
# Enters only during high-liquidity session (08-20 UTC) on volume spikes.
# Long when 4h RSI > 55 (bullish momentum) and price above 1d EMA50 (uptrend).
# Short when 4h RSI < 45 (bearish momentum) and price below 1d EMA50 (downtrend).
# Volume spike (>1.8x 20-bar avg) ensures conviction. Session filter reduces noise.
# Discrete 0.20 position size limits risk. Works in bull/bear via dual RSI/EMA logic.
# Target: 15-30 trades/year (~60-120 total over 4 years) to minimize fee drag.