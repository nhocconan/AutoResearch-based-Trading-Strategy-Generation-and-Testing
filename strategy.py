#!/usr/bin/env python3
"""
1h_4h_1d_TripleConfluence_RSI_Trend_Volume
Hypothesis: On 1h, enter long when RSI < 30 (oversold) with 4h uptrend (EMA50 > EMA200) and 1d volume spike; enter short when RSI > 70 (overbought) with 4h downtrend (EMA50 < EMA200) and 1d volume spike. Exit on opposite RSI cross (50 for long, 50 for short). Uses higher timeframes for trend/volume regime and 1h for precise timing. Designed for low frequency (15-30 trades/year) to avoid fee drag in 1h timeframe.
"""
name = "1h_4h_1d_TripleConfluence_RSI_Trend_Volume"
timeframe = "1h"
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
    
    # 1h RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA50 and EMA200 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    trend_4h = ema_50_4h_aligned > ema_200_4h_aligned  # True for uptrend
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_20avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20avg_1d)
    volume_spike = volume > (2.0 * vol_20avg_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_20avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + 4h uptrend + volume spike
            if rsi[i] < 30 and trend_4h[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + 4h downtrend + volume spike
            elif rsi[i] > 70 and (~trend_4h[i]) and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: RSI crosses back above 50 (for long) or below 50 (for short)
            if position == 1:
                if rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals