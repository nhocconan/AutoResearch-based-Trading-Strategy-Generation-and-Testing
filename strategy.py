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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-day EMA for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 14-day ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 50-day SMA for long-term trend
    sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Volume filter: require volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 50  # need 50 for SMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA20 and SMA50 (uptrend) + high volume + low volatility
            if (close[i] > ema20_1d_aligned[i] and 
                close[i] > sma50_1d_aligned[i] and 
                volume_filter[i] and 
                atr14_1d_aligned[i] < np.nanmedian(atr14_1d_aligned[:i+1])):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA20 and SMA50 (downtrend) + high volume + low volatility
            elif (close[i] < ema20_1d_aligned[i] and 
                  close[i] < sma50_1d_aligned[i] and 
                  volume_filter[i] and 
                  atr14_1d_aligned[i] < np.nanmedian(atr14_1d_aligned[:i+1])):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below EMA20 (trend change) or volatility spike
            if (close[i] < ema20_1d_aligned[i] or 
                atr14_1d_aligned[i] > 2.0 * np.nanmedian(atr14_1d_aligned[:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above EMA20 (trend change) or volatility spike
            if (close[i] > ema20_1d_aligned[i] or 
                atr14_1d_aligned[i] > 2.0 * np.nanmedian(atr14_1d_aligned[:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA20_SMA50_Vol_LowVol_Filter"
timeframe = "1d"
leverage = 1.0