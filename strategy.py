#!/usr/bin/env python3
name = "4h_Stellar_Volume_Target_Signal"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1D data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1D RSI(14)
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan] * 14, rsi_1d[14:]])
    
    # Align RSI
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # 4h Volume spike (current > 1.5x 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma20[:19] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(sma20[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long: RSI oversold (<30), price near lower band, volume spike
        if (position == 0 and 
            rsi_1d_aligned[i] < 30 and 
            close[i] <= lower_band[i] * 1.02 and  # within 2% of lower band
            volume[i] > 1.5 * vol_ma20[i]):
            signals[i] = 0.25
            position = 1
        # Short: RSI overbought (>70), price near upper band, volume spike
        elif (position == 0 and 
              rsi_1d_aligned[i] > 70 and 
              close[i] >= upper_band[i] * 0.98 and  # within 2% of upper band
              volume[i] > 1.5 * vol_ma20[i]):
            signals[i] = -0.25
            position = -1
        elif position == 1:
            # Long exit: RSI > 50 or price crosses above middle band
            if rsi_1d_aligned[i] > 50 or close[i] > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 or price crosses below middle band
            if rsi_1d_aligned[i] < 50 or close[i] < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals