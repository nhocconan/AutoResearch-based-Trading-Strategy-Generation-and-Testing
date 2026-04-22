#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Weekly ATR for volatility regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly ATR
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # 12h ATR for entry trigger and stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h SMA for trend direction
    sma = pd.Series(close).rolling(window=30, min_periods=30).mean().values
    
    # Align weekly ATR and its MA to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_ma_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        sma_val = sma[i]
        atr_1w = atr_1w_aligned[i]
        atr_ma_1w = atr_ma_1w_aligned[i]
        
        # Volatility regime: only trade when weekly ATR is elevated (trending market)
        vol_regime = atr_1w > atr_ma_1w
        
        if position == 0 and vol_regime:
            # Long: price breaks above SMA + 1.5*ATR with rising volatility
            if price > sma_val + 1.5 * atr_val and atr_val > atr[i-1]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below SMA - 1.5*ATR with rising volatility
            elif price < sma_val - 1.5 * atr_val and atr_val > atr[i-1]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: mean reversion to SMA or volatility collapse
            mean_rev = (position == 1 and price < sma_val) or (position == -1 and price > sma_val)
            vol_collapse = atr_val < 0.5 * atr[i-1]  # Sharp drop in volatility
            
            if mean_rev or vol_collapse:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyATRTrend_FilteredBreakout_v1"
timeframe = "12h"
leverage = 1.0