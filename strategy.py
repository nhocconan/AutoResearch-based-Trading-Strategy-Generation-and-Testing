#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(20) for trend direction
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 1d ATR(10) for Keltner channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(10)
    atr_10_1d = pd.Series(tr).ewm(alpha=1/10, adjust=False).mean().values
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Calculate 1d average volume for spike detection (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate Keltner channels from previous 1d OHLC
    # Upper = EMA(20) + 2 * ATR(10)
    # Lower = EMA(20) - 2 * ATR(10)
    keltner_upper = ema_20_1d + 2 * atr_10_1d
    keltner_lower = ema_20_1d - 2 * atr_10_1d
    
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or
            np.isnan(keltner_lower_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_spike = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]  # 50% above average
        
        price = close[i]
        
        # Long when price breaks above upper Keltner band in uptrend with volume spike
        long_setup = price > keltner_upper_aligned[i]
        long_trend = price > ema_20_1d_aligned[i]  # Above EMA = uptrend bias
        long_signal = long_setup and long_trend and vol_spike
        
        # Short when price breaks below lower Keltner band in downtrend with volume spike
        short_setup = price < keltner_lower_aligned[i]
        short_trend = price < ema_20_1d_aligned[i]  # Below EMA = downtrend bias
        short_signal = short_setup and short_trend and vol_spike
        
        # Exit when price returns to EMA(20)
        exit_long = price < ema_20_1d_aligned[i]
        exit_short = price > ema_20_1d_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals