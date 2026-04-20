#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data ONCE
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly True Range and ATR(21)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_21_1w = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # Weekly EMA(34)
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_21_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_21_1w)
    
    # Daily price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_21_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_1w_aligned[i]
        atr_val = atr_21_1w_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Weekly ATR-based volatility filter
        vol_filter = atr_val < np.nanpercentile(atr_21_1w_aligned[:i+1], 50)
        
        if position == 0:
            # Long: price above weekly EMA34, low volatility
            if price > ema_34_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34, low volatility
            elif price < ema_34_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly EMA34 or volatility increases
            if price < ema_34_val or atr_val > np.nanpercentile(atr_21_1w_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly EMA34 or volatility increases
            if price > ema_34_val or atr_val > np.nanpercentile(atr_21_1w_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA34_1w_ATRVolatilityFilter"
timeframe = "1d"
leverage = 1.0