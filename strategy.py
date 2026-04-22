#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 4h high/low for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR for volatility filter and stop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # 4h ATR for breakout multiplier
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF data
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        atr_val = atr[i]
        atr_1d = atr_1d_aligned[i]
        atr_ma_1d = atr_ma_1d_aligned[i]
        
        # Volatility regime: trade only when daily ATR is above its MA
        vol_regime = atr_1d > atr_ma_1d
        
        # Entry conditions
        if position == 0 and vol_regime:
            # Long: break above upper Donchian band
            if price > upper:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian band
            elif price < lower:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position != 0:
            # Exit on Donchian middle line cross
            mid = (upper + lower) / 2
            if (position == 1 and price < mid) or (position == -1 and price > mid):
                signals[i] = 0.0
                position = 0
            # Exit on volatility collapse
            elif atr_val < 0.5 * atr[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_1dATRFilter_VolRegime_v1"
timeframe = "4h"
leverage = 1.0