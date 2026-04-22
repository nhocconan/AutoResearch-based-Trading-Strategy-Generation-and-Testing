#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d data for higher timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align daily ATR and its MA to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h ATR for entry trigger
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h SMA for trend direction
    sma = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(sma[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            continue
        
        price = close[i]
        atr_val = atr[i]
        sma_val = sma[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        atr_1d = atr_1d_aligned[i]
        atr_ma_1d = atr_ma_1d_aligned[i]
        
        # Volatility regime: only trade when daily ATR is elevated (trending market)
        vol_regime = atr_1d > atr_ma_1d
        
        if position == 0 and vol_regime:
            # Long: price breaks above Donchian high + 0.5*ATR with rising volatility
            if price > donchian_high_val + 0.5 * atr_val and atr_val > atr[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low - 0.5*ATR with rising volatility
            elif price < donchian_low_val - 0.5 * atr_val and atr_val > atr[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or volatility collapse
            if price < donchian_low_val or atr_val < 0.5 * atr[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or volatility collapse
            if price > donchian_high_val or atr_val < 0.5 * atr[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dATRFilter_VolRegime_v2"
timeframe = "4h"
leverage = 1.0