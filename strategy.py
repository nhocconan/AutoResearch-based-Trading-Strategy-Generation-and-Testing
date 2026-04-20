#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE (1d and 1w)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ATR on 1d for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 10-period ATR on 1w for weekly volatility
    tr1w = high_1w[1:] - low_1w[1:]
    tr2w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3w = np.abs(low_1w[1:] - close_1w[:-1])
    trw = np.concatenate([[np.nan], np.maximum(tr1w, np.maximum(tr2w, tr3w))])
    atr_10_1w = pd.Series(trw).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d EMA21 for trend
    close_1d_series = pd.Series(close_1d)
    ema_21_1d = close_1d_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1w EMA13 for weekly trend
    close_1w_series = pd.Series(close_1w)
    ema_13_1w = close_1w_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 60-period Donchian channel on 6h (for breakout)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: max high over last 60 periods
    donch_upper = np.full(n, np.nan)
    # Lower band: min low over last 60 periods
    donch_lower = np.full(n, np.nan)
    
    for i in range(60, n):
        donch_upper[i] = np.max(high[i-60:i])
        donch_lower[i] = np.min(low[i-60:i])
    
    # Align HTF indicators to 6h
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(ema_13_1w_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_10_1w_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_21_1d_val = ema_21_1d_aligned[i]
        ema_13_1w_val = ema_13_1w_aligned[i]
        atr_1d_val = atr_14_1d_aligned[i]
        atr_1w_val = atr_10_1w_aligned[i]
        price = close[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        
        # Volatility regime filter: avoid high volatility periods
        vol_ratio = atr_1d_val / atr_1w_val if atr_1w_val > 0 else 1.0
        
        if position == 0:
            # Long: Price breaks above Donchian upper + 1d trend up + 1w trend up + low volatility regime
            if (price > upper and 
                ema_21_1d_val > close_1d[i-1] if i > 0 else False and  # 1d EMA rising
                ema_13_1w_val > close_1w[i-1] if i > 0 else False and  # 1w EMA rising
                vol_ratio < 1.5):  # Not excessively volatile relative to weekly
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + 1d trend down + 1w trend down + low volatility regime
            elif (price < lower and 
                  ema_21_1d_val < close_1d[i-1] if i > 0 else False and  # 1d EMA falling
                  ema_13_1w_val < close_1w[i-1] if i > 0 else False and  # 1w EMA falling
                  vol_ratio < 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian lower OR trend turns against
            if price < lower or ema_21_1d_val < close_1d[i-1] if i > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian upper OR trend turns against
            if price > upper or ema_21_1d_val > close_1d[i-1] if i > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian60_1d1wEMA_TrendFilter"
timeframe = "6h"
leverage = 1.0