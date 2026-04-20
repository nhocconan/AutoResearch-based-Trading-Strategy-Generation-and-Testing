#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) with proper min_periods
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain_ma / loss_ma.replace(0, np.nan)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = rsi_4h.fillna(50).values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h ATR(14) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_4h_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_4h_val = atr_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(adx_val) or 
            np.isnan(atr_4h_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) with strong trend (ADX > 25)
            if rsi_val < 30 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) with strong trend (ADX > 25)
            elif rsi_val > 70 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 or ATR-based stop
            if rsi_val > 50 or close_val < prices['high'].iloc[i] - 1.5 * atr_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 or ATR-based stop
            if rsi_val < 50 or close_val > prices['low'].iloc[i] + 1.5 * atr_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_RSI_ADX_TrendFilter_V1
# Uses 4h RSI for mean reversion signals and 1d ADX for trend strength filter
# Enters long when RSI < 30 and ADX > 25 (oversold in strong trend)
# Enters short when RSI > 70 and ADX > 25 (overbought in strong trend)
# Exits when RSI crosses 50 or ATR-based stop is hit
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_RSI_ADX_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0