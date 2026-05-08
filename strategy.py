#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Pullback_Strategy_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction (primary filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend confirmation (secondary filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d EMA50 for higher timeframe trend confirmation
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h RSI for entry timing (overbought/oversold)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Long conditions: 
        # 1. Price above 4h EMA21 (uptrend)
        # 2. Price above 1d EMA50 (higher timeframe uptrend)
        # 3. RSI < 40 (pullback/oversold)
        # 4. Volume > 1.2x average (confirmation)
        if (position == 0 and
            close[i] > ema_21_4h_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            rsi[i] < 40 and
            vol_ratio[i] > 1.2):
            signals[i] = 0.20
            position = 1
        
        # Short conditions:
        # 1. Price below 4h EMA21 (downtrend)
        # 2. Price below 1d EMA50 (higher timeframe downtrend)
        # 3. RSI > 60 (pullback/overbought)
        # 4. Volume > 1.2x average (confirmation)
        elif (position == 0 and
              close[i] < ema_21_4h_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              rsi[i] > 60 and
              vol_ratio[i] > 1.2):
            signals[i] = -0.20
            position = -1
        
        # Long exit: price breaks below 4h EMA21 OR RSI > 60 (overbought)
        elif position == 1:
            if close[i] < ema_21_4h_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        # Short exit: price breaks above 4h EMA21 OR RSI < 40 (oversold)
        elif position == -1:
            if close[i] > ema_21_4h_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals