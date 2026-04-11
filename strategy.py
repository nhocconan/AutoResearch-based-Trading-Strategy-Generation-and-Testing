#!/usr/bin/env python3
"""
4h_1d_momentum_reversal_v1
Strategy: 4h momentum reversal using RSI divergence and volume confirmation filtered by 1d trend
Timeframe: 4h
Leverage: 1.0
Hypothesis: Captures reversals at RSI extremes (overbought/oversold) with volume confirmation, only in direction of higher timeframe (1d) trend. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend). Target: 20-60 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_momentum_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 1d RSI(14) for trend filter
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(vol_avg[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: 1d RSI > 50 = uptrend, < 50 = downtrend
        uptrend_1d = rsi_1d_aligned[i] > 50
        downtrend_1d = rsi_1d_aligned[i] < 50
        
        # Reversal conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: RSI oversold with volume in uptrend
        long_signal = rsi_oversold and vol_confirmed and uptrend_1d
        
        # Short: RSI overbought with volume in downtrend
        short_signal = rsi_overbought and vol_confirmed and downtrend_1d
        
        # Exit when RSI returns to neutral (40-60)
        exit_long = position == 1 and rsi[i] > 40
        exit_short = position == -1 and rsi[i] < 60
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals