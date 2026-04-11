#!/usr/bin/env python3
"""
12h_1d_rsi_reversion_v1
Strategy: 12h RSI mean reversion with 1-day volatility filter and volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: In ranging markets (1-day ATR ratio < 0.6), RSI extremes (RSI < 30 for long, RSI > 70 for short) on 12h timeframe with volume confirmation (volume > 1.5x 20-period average) provide mean-reversion entries. Works in both bull and bear markets by capturing overextended moves during low volatility regimes. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_rsi_reversion_v1"
timeframe = "12h"
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
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1-day ATR (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1-day ATR ratio: current ATR / 20-period average ATR (low when < 0.6)
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / (atr_ma_20_1d + 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: volume > 1.5x 20-period average
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: low volatility regime (1-day ATR ratio < 0.6)
        low_volatility = atr_ratio_1d_aligned[i] < 0.6
        
        # RSI mean reversion conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Long conditions: RSI oversold + volume confirmation + low volatility
        long_signal = rsi_oversold and volume_confirmed and low_volatility
        
        # Short conditions: RSI overbought + volume confirmation + low volatility
        short_signal = rsi_overbought and volume_confirmed and low_volatility
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and rsi[i] >= 40
        exit_short = position == -1 and rsi[i] <= 60
        
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