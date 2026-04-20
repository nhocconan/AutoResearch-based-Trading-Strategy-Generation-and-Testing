#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-week ATR on weekly data
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-week EMA on weekly data
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 10-week EMA on weekly data
    ema_10_1w = close_1w_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate daily ATR for reference
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr_d1 = high[1:] - low[1:]
    tr_d2 = np.abs(high[1:] - close[:-1])
    tr_d3 = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr_d1, np.maximum(tr_d2, tr_d3))])
    atr_14_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if NaN
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_10_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or np.isnan(atr_14_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_20_1w_val = ema_20_1w_aligned[i]
        ema_10_1w_val = ema_10_1w_aligned[i]
        atr_1w_val = atr_14_1w_aligned[i]
        atr_d_val = atr_14_d[i]
        price = close[i]
        
        if position == 0:
            # Long: Weekly EMA10 > EMA20 (bullish trend) + low volatility regime
            if ema_10_1w_val > ema_20_1w_val and atr_d_val < 0.8 * atr_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: Weekly EMA10 < EMA20 (bearish trend) + low volatility regime
            elif ema_10_1w_val < ema_20_1w_val and atr_d_val < 0.8 * atr_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend reversal or volatility expansion
            if ema_10_1w_val < ema_20_1w_val or atr_d_val > 1.2 * atr_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend reversal or volatility expansion
            if ema_10_1w_val > ema_20_1w_val or atr_d_val > 1.2 * atr_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA10_EMA20_VolatilityFilter"
timeframe = "1d"
leverage = 1.0