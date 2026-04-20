#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d EMA10
    ema_10_1d = close_1d_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 12h EMA21
    close = prices['close'].values
    close_series = pd.Series(close)
    ema_21_12h = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h volume average (20-period)
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    # Align HTF indicators to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_10_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        ema_10_1d_val = ema_10_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        ema_21_val = ema_21_12h[i]
        
        if position == 0:
            # Long: EMA10 > EMA34 (bullish) + volatility filter (low volatility) + volume confirmation
            if ema_10_1d_val > ema_34_1d_val and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 30) and vol_val > 1.2 * vol_ma_val:
                signals[i] = 0.30
                position = 1
            # Short: EMA10 < EMA34 (bearish) + volatility filter (low volatility) + volume confirmation
            elif ema_10_1d_val < ema_34_1d_val and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 30) and vol_val > 1.2 * vol_ma_val:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: EMA10 < EMA34 or volatility spike
            if ema_10_1d_val < ema_34_1d_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: EMA10 > EMA34 or volatility spike
            if ema_10_1d_val > ema_34_1d_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_EMA10_EMA34_VolatilityFilter_Volume"
timeframe = "12h"
leverage = 1.0