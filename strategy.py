#!/usr/bin/env python3
name = "4h_RSI2_Pullback_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 200 EMA for trend filter (daily)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # RSI(2) on daily (14-period RSI but 2-period lookback)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # ATR(14) for pullback measurement
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h RSI(14) for entry timing
    delta_4h = np.diff(close, prepend=close[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = np.where(avg_loss_4h != 0, avg_gain_4h / avg_loss_4h, 0)
    rsi_14_4h = 100 - (100 / (1 + rs_4h))
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(200, 14, 14, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema200_aligned[i]) or np.isnan(rsi_2_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(rsi_14_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend (price > 200 EMA) + daily RSI(2) oversold + 4h RSI(14) pullback + volume spike
            if (close[i] > ema200_aligned[i] and 
                rsi_2_aligned[i] < 10 and 
                rsi_14_4h[i] < 30 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < 200 EMA) + daily RSI(2) overbought + 4h RSI(14) bounce + volume spike
            elif (close[i] < ema200_aligned[i] and 
                  rsi_2_aligned[i] > 90 and 
                  rsi_14_4h[i] > 70 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: daily RSI(2) overbought or 4h RSI(14) overbought
            if rsi_2_aligned[i] > 80 or rsi_14_4h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: daily RSI(2) oversold or 4h RSI(14) oversold
            if rsi_2_aligned[i] < 20 or rsi_14_4h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals