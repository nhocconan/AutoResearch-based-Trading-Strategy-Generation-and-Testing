#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_RSI_Momentum_v1
Hypothesis: On 4h timeframe, use 1d KAMA to determine trend direction, then enter long when RSI(14) crosses above 30 and short when RSI(14) crosses below 70, with momentum confirmation via ROC(10). This avoids overtrading by requiring both trend alignment and momentum confirmation. Designed for 20-50 trades/year by using higher timeframe trend filter and momentum thresholds. Works in bull markets via KAMA-up/RSI-long and in bear markets via KAMA-down/RSI-short.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_RSI_Momentum_v1"
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
    
    # Volume average (20 period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum |diff| over 10 periods
    # Fix dimensions: change length = len-10, volatility length = len-1
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    # Align to same length
    min_len = min(len(change_padded), len(volatility_padded))
    change_padded = change_padded[:min_len]
    volatility_padded = volatility_padded[:min_len]
    # Pad to full length
    er = np.full(len(close_1d), np.nan)
    er[10:] = change_padded[10:] / volatility_padded[10:]
    er = np.where(volatility_padded[10:] == 0, 0, er[10:])
    er = np.concatenate([np.full(10, np.nan), er])
    # Smoothing constants
    sc = (er * 0.29 + 0.06) ** 2  # where 0.29 = 2/(2+1), 0.06 = 2/(30+1)
    # KAMA calculation
    kama = np.full(len(close_1d), np.nan)
    kama[9] = close_1d[9]  # start at index 9
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ROC(10) for momentum confirmation
    roc = np.zeros_like(close)
    roc[:10] = np.nan
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(roc[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition: price relative to KAMA
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI conditions with momentum confirmation
        rsi_oversold = rsi[i] < 30 and roc[i] > 0  # RSI below 30 with positive momentum
        rsi_overbought = rsi[i] > 70 and roc[i] < 0  # RSI above 70 with negative momentum
        
        # Volume confirmation: current volume > 1.2x average
        volume_confirm = volume[i] > vol_ma[i] * 1.2
        
        # Entry conditions
        long_entry = price_above_kama and rsi_oversold and volume_confirm
        short_entry = price_below_kama and rsi_overbought and volume_confirm
        
        # Exit conditions: opposite RSI extreme
        long_exit = rsi[i] > 70
        short_exit = rsi[i] < 30
        
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