#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_200EMA_AboveVolumeBreakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for 200 EMA and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Daily 200 EMA for trend filter
    ema_200_d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_d)
    
    # Daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_d_aligned = align_htf_to_ltf(prices, df_1d, atr_d)
    
    # 4h volume filter: current volume > 2.0 * 24-period average (24 * 4h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(atr_d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_200_val = ema_200_aligned[i]
        atr_val = atr_d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price above 200 EMA with volume and sufficient volatility
            if close_val > ema_200_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: price falls back below 200 EMA
            if close_val < ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals