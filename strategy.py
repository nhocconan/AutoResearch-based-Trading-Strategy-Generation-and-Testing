#!/usr/bin/env python3
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
    
    # Get 1d data for daily EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 12h data for volume spike detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume MA20
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Get 4h data for RSI calculation
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(200, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_12h_aligned[i] if vol_ma_20_12h_aligned[i] > 0 else 0
        
        # Determine trend from daily EMA200
        uptrend = price > ema_200_1d_aligned[i]
        downtrend = price < ema_200_1d_aligned[i]
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_extreme = (rsi[i] > 30) and (rsi[i] < 70)
        
        if position == 0:
            # Long entry: price above daily EMA200, volume spike, RSI not extreme
            if uptrend and volume_confirmation and rsi_not_extreme:
                signals[i] = size
                position = 1
            # Short entry: price below daily EMA200, volume spike, RSI not extreme
            elif downtrend and volume_confirmation and rsi_not_extreme:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below daily EMA200 or RSI overbought
            if price < ema_200_1d_aligned[i] or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price above daily EMA200 or RSI oversold
            if price > ema_200_1d_aligned[i] or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA200_Trend_Volume_RSI_Filter"
timeframe = "4h"
leverage = 1.0