#!/usr/bin/env python3
"""
4h_1d_MeanReversion_RSI_with_Regime_Filter_v1
Hypothesis: 4h mean reversion using RSI extremes with 1d volatility regime filter.
In bull markets, buy RSI<30 when 1d volatility is low (mean reversion works).
In bear markets, sell RSI>70 when 1d volatility is high (trend exhaustion).
Volatility filter prevents whipsaw in strong trends. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_MeanReversion_RSI_with_Regime_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for RSI and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d closes
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # Not enough data
    
    # Align RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d volatility (ATR ratio regime filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:15])
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    atr_1d[:13] = np.nan
    
    # ATR ratio: current ATR / 50-period MA of ATR (volatility regime)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 4h RSI for entry timing (more responsive)
    delta_4h = np.diff(close, prepend=close[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = np.zeros_like(gain_4h)
    avg_loss_4h = np.zeros_like(loss_4h)
    avg_gain_4h[13] = np.mean(gain_4h[1:14])
    avg_loss_4h[13] = np.mean(loss_4h[1:14])
    
    for i in range(14, len(gain_4h)):
        avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
        avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    
    rs_4h = np.where(avg_loss_4h != 0, avg_gain_4h / avg_loss_4h, 100)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h[:13] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi_4h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: volatility state
        high_vol = atr_ratio_aligned[i] > 1.2  # High volatility regime
        low_vol = atr_ratio_aligned[i] < 0.8   # Low volatility regime
        
        # Mean reversion signals
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        
        # Entry logic: regime-dependent mean reversion
        long_entry = rsi_oversold and low_vol  # Buy oversold in low vol (bull mean reversion)
        short_entry = rsi_overbought and high_vol  # Sell overbought in high vol (bear exhaustion)
        
        # Exit: RSI returns to neutral zone
        long_exit = rsi_4h[i] > 50
        short_exit = rsi_4h[i] < 50
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals