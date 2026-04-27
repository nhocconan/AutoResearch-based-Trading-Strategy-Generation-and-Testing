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
    
    # Get daily data for calculations (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily RSI(14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_vals = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_vals)
    
    # Daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 6h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema50_6h = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(ema50_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price above EMA50 + high volume
            if (rsi_1d_aligned[i] < 30 and 
                close[i] > ema50_6h[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price below EMA50 + high volume
            elif (rsi_1d_aligned[i] > 70 and 
                  close[i] < ema50_6h[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or price crosses below EMA50
            if rsi_1d_aligned[i] > 50 or close[i] < ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or price crosses above EMA50
            if rsi_1d_aligned[i] < 50 or close[i] > ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI14_Daily_OverboughtOversold_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0