#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop (1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily ADX(14) for trend strength
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate daily RSI(14) for overbought/oversold
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        adx_val = adx_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(rsi_val) or 
            np.isnan(atr_val) or np.isnan(upper) or np.isnan(lower)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: strong trend (ADX > 25), oversold (RSI < 30), price at lower Donchian
            if adx_val > 25 and rsi_val < 30 and close_val <= lower:
                signals[i] = 0.25
                position = 1
            # Short: strong trend (ADX > 25), overbought (RSI > 70), price at upper Donchian
            elif adx_val > 25 and rsi_val > 70 and close_val >= upper:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakening (ADX < 20) or overbought (RSI > 70) or price at upper Donchian
            if adx_val < 20 or rsi_val > 70 or close_val >= upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakening (ADX < 20) or oversold (RSI < 30) or price at lower Donchian
            if adx_val < 20 or rsi_val < 30 or close_val <= lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_ADX_RSI_Donchian
# Uses daily ADX(14) for trend strength filter
# Enters long when ADX > 25, RSI < 30, price touches lower Donchian(20)
# Enters short when ADX > 25, RSI > 70, price touches upper Donchian(20)
# Exits when ADX < 20, RSI reaches opposite extreme, or price touches opposite band
# Designed for 12h timeframe with ~15-25 trades/year
name = "12h_ADX_RSI_Donchian"
timeframe = "12h"
leverage = 1.0