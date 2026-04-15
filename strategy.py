#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily Bollinger Bands(20,2) for mean reversion signals
    sma_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_bb_1d_aligned[i]) or np.isnan(lower_bb_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Long signal: price touches or breaks below lower BB in high vol (mean reversion long)
        # Short signal: price touches or breaks above upper BB in high vol (mean reversion short)
        if vol_filter:
            if close[i] <= lower_bb_1d_aligned[i]:
                signals[i] = 0.25  # long 25%
            elif close[i] >= upper_bb_1d_aligned[i]:
                signals[i] = -0.25  # short 25%
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_BollingerMeanReversion_VolFilter_v1"
timeframe = "4h"
leverage = 1.0