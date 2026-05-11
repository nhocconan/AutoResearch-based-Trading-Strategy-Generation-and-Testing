# 4h_Momentum_Volume_Trend
# Hypothesis: Combines 4h momentum (RSI > 55) with volume confirmation (volume > 1.5x 20-period average) and trend filter (price > 4h EMA50) for long entries.
# Short entries use RSI < 45, volume confirmation, and price < 4h EMA50.
# Works in both bull and bear markets by capturing momentum shifts with volume confirmation.
# Uses 4h timeframe to limit trades (target: 20-50/year) and reduce fee drag.
# Exit when momentum reverses (RSI crosses back below 50 for longs, above 50 for shorts).

#!/usr/bin/env python3
name = "4h_Momentum_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI calculation on 4h closes
    delta = np.diff(df_4h['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Volume ratio: current volume / 20-period average
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_4h['volume'].values / vol_ma_4h
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of EMA50, RSI, vol lookback)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum), volume surge, price above EMA50
            if (rsi_aligned[i] > 55 and 
                vol_ratio_aligned[i] > 1.5 and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45 (bearish momentum), volume surge, price below EMA50
            elif (rsi_aligned[i] < 45 and 
                  vol_ratio_aligned[i] > 1.5 and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: momentum reversal (RSI crosses back below/above 50)
            if position == 1:
                if rsi_aligned[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if rsi_aligned[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals