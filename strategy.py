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
    
    # Get 1d data for trend filter and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d ATR20 for volatility filter (only trade in high volatility)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR20
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr20_aligned = align_htf_to_ltf(prices, df_1d, atr20)
    
    # Current 12h ATR for dynamic sizing
    tr_12h1 = high - low
    tr_12h2 = np.abs(high - np.roll(close, 1))
    tr_12h3 = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when 1d ATR20 > its 50-period average (high vol regime)
    atr20_ma = pd.Series(atr20_aligned).rolling(window=50, min_periods=50).mean().values
    high_vol_regime = atr20_aligned > atr20_ma
    
    # Session filter: 08-20 UTC (active trading hours)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 50  # need 50 for EMA50 and ATR calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr20_aligned[i]) or 
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend + high volatility + session
            if (close[i] > ema50_1d_aligned[i] and 
                high_vol_regime[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + high volatility + session
            elif (close[i] < ema50_1d_aligned[i] and 
                  high_vol_regime[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend reversal or low volatility
            if (close[i] < ema50_1d_aligned[i] or not high_vol_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or low volatility
            if (close[i] > ema50_1d_aligned[i] or not high_vol_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA50_Trend_HighVol_Session"
timeframe = "12h"
leverage = 1.0