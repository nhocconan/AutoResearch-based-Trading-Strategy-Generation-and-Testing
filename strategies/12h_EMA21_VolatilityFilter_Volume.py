#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h close EMA21 for trend
    close = prices['close'].values
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h volume average (20-period)
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        ema_fast = ema_21[i]
        
        if position == 0:
            # Long: price > EMA21 (uptrend) + volatility filter (low volatility) + volume confirmation
            if price > ema_fast and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 30) and vol_val > 1.2 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: price < EMA21 (downtrend) + volatility filter + volume confirmation
            elif price < ema_fast and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 30) and vol_val > 1.2 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < EMA21 or volatility spike
            if price < ema_fast or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > EMA21 or volatility spike
            if price > ema_fast or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA21_VolatilityFilter_Volume"
timeframe = "12h"
leverage = 1.0