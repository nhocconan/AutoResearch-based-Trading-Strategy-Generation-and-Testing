#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1w RSI (14) for long-term momentum
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(avg_loss == 0, 100, rsi_1w)
    rsi_1w = np.where(avg_gain == 0, 0, rsi_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 1d RSI (14) for short-term momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.where(avg_loss == 0, 100, rsi_1d)
    rsi_1d = np.where(avg_gain == 0, 0, rsi_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ADX (14) for trend strength
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
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    up14 = pd.Series(up_move).rolling(window=14, min_periods=14).mean().values
    down14 = pd.Series(down_move).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * np.divide(up14, tr14, out=np.zeros_like(up14), where=tr14!=0)
    minus_di = 100 * np.divide(down14, tr14, out=np.zeros_like(down14), where=tr14!=0)
    
    # DX and ADX
    dx = np.divide(np.abs(plus_di - minus_di), (plus_di + minus_di), out=np.zeros_like(plus_di), where=(plus_di + minus_di)!=0) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_1w_val = rsi_1w_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_1w_val) or np.isnan(rsi_1d_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI weekly > 50 (bullish momentum) and RSI daily < 30 (oversold) and ADX > 25 (trending)
            if rsi_1w_val > 50 and rsi_1d_val < 30 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: RSI weekly < 50 (bearish momentum) and RSI daily > 70 (overbought) and ADX > 25 (trending)
            elif rsi_1w_val < 50 and rsi_1d_val > 70 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI daily > 70 (overbought) or ADX < 20 (weak trend)
            if rsi_1d_val > 70 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI daily < 30 (oversold) or ADX < 20 (weak trend)
            if rsi_1d_val < 30 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_1wRSI_1dRSI_ADXFilter_V1
# Uses 1-week RSI for long-term momentum direction and 1-day RSI for overbought/oversold signals
# Enters long when weekly RSI > 50 (bullish momentum) AND daily RSI < 30 (oversold) AND ADX > 25 (strong trend)
# Enters short when weekly RSI < 50 (bearish momentum) AND daily RSI > 70 (overbought) AND ADX > 25 (strong trend)
# Exits when daily RSI reaches opposite extreme or ADX weakens (<20)
# Designed for 12h timeframe with ~12-37 trades/year
name = "12h_1wRSI_1dRSI_ADXFilter_V1"
timeframe = "12h"
leverage = 1.0