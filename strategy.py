#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data (primary) and 12h data (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA on 4h data (21-period)
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate RSI on 12h data (14-period)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_12h = 100 - (100 / (1 + rs))
    
    # Calculate volume ratio on 12h data
    volume_12h = df_12h['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi_14_12h_aligned[i]) or
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x 20-period average
        volume_condition = volume_12h[i // 2] > (volume_ma_20_12h_aligned[i] * 1.5)
        
        # Trend filter: EMA direction on 4h
        # Long when close > EMA21 (uptrend)
        # Short when close < EMA21 (downtrend)
        long_trend = close[i] > ema_21_4h_aligned[i]
        short_trend = close[i] < ema_21_4h_aligned[i]
        
        # Momentum filter: RSI extremes on 12h
        # Long when RSI < 30 (oversold)
        # Short when RSI > 70 (overbought)
        rsi_oversold = rsi_14_12h_aligned[i] < 30
        rsi_overbought = rsi_14_12h_aligned[i] > 70
        
        if position == 0:
            if long_trend and volume_condition and rsi_oversold:
                position = 1
                signals[i] = position_size
            elif short_trend and volume_condition and rsi_overbought:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when trend reverses or RSI becomes overbought
            if not long_trend or rsi_14_12h_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when trend reverses or RSI becomes oversold
            if not short_trend or rsi_14_12h_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_EMA21_RSI14_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0