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
    
    # Get daily data for trend filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # 6-hour volatility regime: current ATR(6) vs daily ATR(14)
    # Calculate 6h ATR
    tr_6h1 = high - low
    tr_6h2 = np.abs(high - np.roll(close, 1))
    tr_6h3 = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr6 = pd.Series(tr_6h).rolling(window=6, min_periods=6).mean().values
    
    # Volatility regime: low volatility when 6h ATR < 0.7 * daily ATR
    vol_regime = atr6 < (0.7 * atr14_aligned)
    
    # Volume filter: require volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.2)
    
    # Session filter: 08-20 UTC (active trading hours)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 34  # need 34 for EMA34 and ATR14
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr6[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Low volatility regime + price > EMA34 (bullish bias) + volume + session
            if (vol_regime[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Low volatility regime + price < EMA34 (bearish bias) + volume + session
            elif (vol_regime[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: volatility expansion OR price < EMA34 (trend change)
            if (not vol_regime[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: volatility expansion OR price > EMA34 (trend change)
            if (not vol_regime[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolRegime_EMA34_Bias_Volume_Session"
timeframe = "6h"
leverage = 1.0