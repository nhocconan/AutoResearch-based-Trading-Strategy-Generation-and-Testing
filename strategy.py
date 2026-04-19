#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_TRIX_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for TRIX calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # TRIX (15-period) - Triple Exponential Average derivative
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) then rate of change
    close_series = pd.Series(df_1w['close'])
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change() * 100  # percentage change
    trix = trix_raw.values
    
    # Align TRIX to daily timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Volume spike: current volume > 2.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period weekly)
    tr = np.maximum(df_1w['high'].values[1:] - df_1w['low'].values[1:], 
                    np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1]))
    tr = np.maximum(tr, np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        trix_val = trix_aligned[i]
        atr = atr_14_1w_aligned[i]
        
        volume_spike = vol > 2.5 * vol_ma
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike
            if trix_val > 0 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike
            elif trix_val < 0 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero or volatility drop
            if trix_val < 0 or vol < vol_ma * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero or volatility drop
            if trix_val > 0 or vol < vol_ma * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals