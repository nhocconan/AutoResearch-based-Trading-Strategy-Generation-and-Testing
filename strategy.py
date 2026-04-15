#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume confirmation and 1d ATR-based volatility filter.
# In bull markets: breaks above R3 capture momentum; in bear markets: breaks below S3 capture continuation.
# Volatility filter avoids chop, volume confirmation reduces false breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) with discrete sizing 0.25 to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prior_high - prior_low)
    camarilla_s3 = camarilla_pivot - 1.1 * (prior_high - prior_low)
    
    # Align Camarilla levels to 12h
    camarilla_pivot_12h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(camarilla_pivot_12h[i]) or 
            np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.6% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions: price above pivot, breaks above R3 with volume, volatility regime
        if (close[i] > camarilla_pivot_12h[i] and
            close[i] > camarilla_r3_12h[i] and
            volume_ratio[i] > 1.5 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions: price below pivot, breaks below S3 with volume, volatility regime
        elif (close[i] < camarilla_pivot_12h[i] and
              close[i] < camarilla_s3_12h[i] and
              volume_ratio[i] > 1.5 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Vol_Regime_Camarilla_Pivot_R3S3_Breakout_v3"
timeframe = "12h"
leverage = 1.0