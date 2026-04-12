#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14) for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR(14) for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Bollinger Bands (20, 2)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Align daily indicators to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # 4h Bollinger Bands for entry timing
    sma_20_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma_20_4h + 2 * std_20_4h
    lower_bb_4h = sma_20_4h - 2 * std_20_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_bb_4h[i]) or np.isnan(lower_bb_4h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price touches Bollinger Band + RSI extreme + EMA trend
        long_condition = (close[i] <= lower_bb_4h[i] and 
                         rsi_1d_aligned[i] < 30 and 
                         close[i] > ema_50_1d_aligned[i])
        short_condition = (close[i] >= upper_bb_4h[i] and 
                          rsi_1d_aligned[i] > 70 and 
                          close[i] < ema_50_1d_aligned[i])
        
        # Exit conditions: mean reversion back to middle band or trend change
        middle_bb_4h = sma_20_4h[i]
        long_exit = close[i] >= middle_bb_4h
        short_exit = close[i] <= middle_bb_4h
        
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_bb_rsi_ema_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0