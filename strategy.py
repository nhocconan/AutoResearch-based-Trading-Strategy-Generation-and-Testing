#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1d ATR for volatility regime filter
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
    atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    
    # 4h Bollinger Bands (20, 2.0) for mean reversion
    close = prices['close'].values
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2.0 * std_20
    lower_bb = ma_20 - 2.0 * std_20
    
    # Align daily ATR and its MA to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(ma_20[i]) or 
            np.isnan(std_20[i]) or
            np.isnan(upper_bb[i]) or
            np.isnan(lower_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ma_20_val = ma_20[i]
        upper_bb_val = upper_bb[i]
        lower_bb_val = lower_bb[i]
        atr_1d = atr_1d_aligned[i]
        atr_ma_1d = atr_ma_1d_aligned[i]
        
        # Volatility regime: only trade when daily ATR is elevated (trending market)
        vol_regime = atr_1d > atr_ma_1d
        
        if position == 0 and vol_regime:
            # Long: price touches lower BB
            if price <= lower_bb_val:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: price touches upper BB
            elif price >= upper_bb_val:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: mean reversion to middle band
            mean_rev = (position == 1 and price >= ma_20_val) or (position == -1 and price <= ma_20_val)
            
            if mean_rev:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_BollingerBandsMeanReversion_1dVolRegime_v1"
timeframe = "4h"
leverage = 1.0