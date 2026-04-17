# 4h_RSI14_1dADX25_Volume1.5x
# Hypothesis: RSI mean reversion on 4h with 1d ADX trend filter and volume confirmation.
# Works in bull/bear: RSI identifies overextended moves, ADX ensures trading with trend,
# volume confirms institutional interest. Target 20-50 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === 4h RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d ADX(14) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / (tr14 + 1e-10)
    minus_di = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1d volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume and align
        df_1d_current = get_htf_data(prices, '1d')
        volume_1d_current = df_1d_current['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d_current, volume_1d_current)
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        vol_spike = volume_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # ADX trend filter
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI oversold + volume spike + trending
            if rsi_oversold and vol_spike and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI overbought + volume spike + trending
            elif rsi_overbought and vol_spike and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when RSI returns to neutral or trend weakens
            if rsi_neutral or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI returns to neutral or trend weakens
            if rsi_neutral or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI14_1dADX25_Volume1.5x"
timeframe = "4h"
leverage = 1.0