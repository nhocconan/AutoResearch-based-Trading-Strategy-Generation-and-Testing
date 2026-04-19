#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily close for TRIX calculation
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: 15-period EMA applied 3 times
    close_series = pd.Series(close_1d)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    
    # TRIX = (EMA3 - EMA3_previous) / EMA3_previous * 100
    ema3_lag1 = np.roll(ema3.values, 1)
    ema3_lag1[0] = np.nan
    trix_raw = (ema3.values - ema3_lag1) / ema3_lag1 * 100
    
    # Align TRIX to 4h timeframe
    trix_1d = trix_raw
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Daily ATR for volatility filter (14-period)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current volume > 2.0x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: price above/below 50-period EMA (4h)
    close_series_4h = pd.Series(close)
    ema_50 = close_series_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        trix = trix_1d_aligned[i]
        atr = atr_14_aligned[i]
        ema = ema_50[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        # TRIX momentum: positive for long, negative for short
        trix_long = trix > 0
        trix_short = trix < 0
        
        if position == 0:
            # Long: TRIX positive, volume spike, price above EMA
            if trix_long and volume_confirmed and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative, volume spike, price below EMA
            elif trix_short and volume_confirmed and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX turns negative or price below EMA
            if trix < 0 or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX turns positive or price above EMA
            if trix > 0 or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals