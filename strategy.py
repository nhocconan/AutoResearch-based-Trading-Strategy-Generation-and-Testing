#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d ATR for volatility regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for daily ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # 1h ATR for entry trigger
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h SMA for trend direction
    sma = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Align daily ATR and its MA to 1h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Precompute session mask (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        sma_val = sma[i]
        atr_1d = atr_1d_aligned[i]
        atr_ma_1d = atr_ma_1d_aligned[i]
        
        # Volatility regime: only trade when daily ATR is elevated (trending market)
        vol_regime = atr_1d > atr_ma_1d
        
        if position == 0 and vol_regime:
            # Long: price breaks above SMA + 1.5*ATR with rising volatility
            if price > sma_val + 1.5 * atr_val and atr_val > atr[i-1]:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below SMA - 1.5*ATR with rising volatility
            elif price < sma_val - 1.5 * atr_val and atr_val > atr[i-1]:
                signals[i] = -0.20
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_1dATRRegime_SMA_RiseBreakout_v1"
timeframe = "1h"
leverage = 1.0