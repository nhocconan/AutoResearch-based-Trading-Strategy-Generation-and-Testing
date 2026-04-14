#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for 1d indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12-period EMA on daily close
    close_1d_series = pd.Series(close_1d)
    ema_12_1d = close_1d_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate daily ATR (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily indicators to 12h timeframe
    ema_12_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_12_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-period EMA on 12h close
    close_series = pd.Series(close)
    ema_12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(12, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12_1d_aligned[i]) or 
            np.isnan(ema_12[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_1d_aligned[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above 12h EMA AND daily EMA rising
            if (close[i] > ema_12[i] and 
                ema_12_1d_aligned[i] > ema_12_1d_aligned[i-1]):
                position = 1
                signals[i] = position_size
            # Short: Price below 12h EMA AND daily EMA falling
            elif (close[i] < ema_12[i] and 
                  ema_12_1d_aligned[i] < ema_12_1d_aligned[i-1]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below 12h EMA
            if close[i] < ema_12[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above 12h EMA
            if close[i] > ema_12[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_EMA12_Trend"
timeframe = "12h"
leverage = 1.0