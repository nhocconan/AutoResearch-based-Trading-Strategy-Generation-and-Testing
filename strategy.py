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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime filter
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
    camarilla_r4 = camarilla_pivot + 1.5 * (prior_high - prior_low)
    camarilla_s4 = camarilla_pivot - 1.5 * (prior_high - prior_low)
    
    # Align Camarilla levels to 1d
    camarilla_pivot_1d = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(camarilla_pivot_1d[i]) or 
            np.isnan(camarilla_r3_1d[i]) or np.isnan(camarilla_s3_1d[i]) or 
            np.isnan(camarilla_r4_1d[i]) or np.isnan(camarilla_s4_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. Price above Camarilla pivot (bullish bias)
        # 2. Price breaks above Camarilla R3 with volume confirmation
        if (close[i] > camarilla_pivot_1d[i] and
            close[i] > camarilla_r3_1d[i] and
            volume[i] > volume[i-1] and  # volume increasing
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below Camarilla pivot (bearish bias)
        # 2. Price breaks below Camarilla S3 with volume confirmation
        elif (close[i] < camarilla_pivot_1d[i] and
              close[i] < camarilla_s3_1d[i] and
              volume[i] > volume[i-1] and  # volume increasing
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Vol_Regime_Camarilla_Pivot_R3S3_Breakout_v2"
timeframe = "1d"
leverage = 1.0