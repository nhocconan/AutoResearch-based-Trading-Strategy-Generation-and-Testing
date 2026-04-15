#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h Camarilla pivot levels (using prior 12h bar's OHLC)
    prior_high = df_12h['high'].shift(1).values
    prior_low = df_12h['low'].shift(1).values
    prior_close = df_12h['close'].shift(1).values
    
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prior_high - prior_low)
    camarilla_s3 = camarilla_pivot - 1.1 * (prior_high - prior_low)
    
    # Align Camarilla levels to 4h
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_12h_aligned[i]) or np.isnan(camarilla_pivot_4h[i]) or 
            np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 12h ATR is elevated (> 0.8% of price)
        # This avoids low-volatility chop and focuses on momentum/trend days
        vol_regime = atr_14_12h_aligned[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. Price above Camarilla pivot (bullish bias)
        # 2. Price breaks above Camarilla R3 with volume (bullish continuation)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Daily volatility regime filter (avoid chop)
        if (close[i] > camarilla_pivot_4h[i] and
            close[i] > camarilla_r3_4h[i] and
            volume_ratio[i] > 1.8 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below Camarilla pivot (bearish bias)
        # 2. Price breaks below Camarilla S3 with volume (bearish continuation)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Daily volatility regime filter
        elif (close[i] < camarilla_pivot_4h[i] and
              close[i] < camarilla_s3_4h[i] and
              volume_ratio[i] > 1.8 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Vol_Regime_Camarilla_Pivot_R3S3_Breakout_v3"
timeframe = "4h"
leverage = 1.0