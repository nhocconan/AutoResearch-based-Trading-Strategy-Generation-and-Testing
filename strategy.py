#!/usr/bin/env python3
"""
1h_4d_Swing_Reversal
Hypothesis: Identify swing reversals on 1h using 4h/1d trend context and volume confirmation.
In bull markets: buy pullbacks in uptrend; in bear markets: sell rallies in downtrend.
Uses 4h EMA for trend direction, 1d RSI for overbought/oversold conditions, and volume spikes for confirmation.
Designed for low trade frequency (15-30/year) to minimize fee drag while capturing meaningful swings.
Works in both bull (buy dips) and bear (sell rallies) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for RSI (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI manually
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike: current volume > 2.0 x 20-period average (conservative)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # Fixed size to minimize churn
    
    for i in range(100, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long condition: price above 4h EMA (uptrend), RSI oversold (<30), volume spike
        long_condition = (close[i] > ema_4h_aligned[i]) and (rsi_1d_aligned[i] < 30) and volume_spike[i]
        
        # Short condition: price below 4h EMA (downtrend), RSI overbought (>70), volume spike
        short_condition = (close[i] < ema_4h_aligned[i]) and (rsi_1d_aligned[i] > 70) and volume_spike[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4d_Swing_Reversal"
timeframe = "1h"
leverage = 1.0