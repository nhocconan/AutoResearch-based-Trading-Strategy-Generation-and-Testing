#!/usr/bin/env python3
"""
Hypothesis: 12h 1-week RSI momentum with 1d volume confirmation and volatility regime.
Long when weekly RSI > 55 (bullish momentum) + volume spike + low volatility (ATR ratio < 0.8).
Short when weekly RSI < 45 (bearish momentum) + volume spike + low volatility.
Uses 1-week RSI for momentum, 1d volume spike (volume > 1.5x 20-period average) for confirmation,
and 1-week volatility regime (ATR ratio < 0.8) to avoid false signals in high volatility.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get 1w data for RSI and volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1-week RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first 14 periods average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[:13] = np.nan  # first 13 values undefined
    
    # Calculate 1-week ATR for volatility regime
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-week ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma_50 = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1w / atr_ma_50
    
    # Volatility regime: ATR ratio < 0.8 = low volatility (good for signals)
    low_volatility = atr_ratio < 0.8
    
    # Align all 1w indicators to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))  # already done above
    low_volatility_aligned = align_htf_to_ltf(prices, df_1w, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(low_volatility_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: RSI momentum + volume spike + low volatility
        rsi_bullish = rsi_1w_aligned[i] > 55
        rsi_bearish = rsi_1w_aligned[i] < 45
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        vol_regime = low_volatility_aligned[i] > 0.5  # True if low volatility
        
        long_entry = rsi_bullish and vol_confirm and vol_regime
        short_entry = rsi_bearish and vol_confirm and vol_regime
        
        # Exit when RSI returns to neutral zone (45-55)
        exit_long = position == 1 and (rsi_1w_aligned[i] < 55)
        exit_short = position == -1 and (rsi_1w_aligned[i] > 45)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_rsi_vol_volatility"
timeframe = "12h"
leverage = 1.0