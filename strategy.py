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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14) for mean reversion signals
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_14 = -100 * (highest_high_14 - df_1d['close'].values) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_14_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14)
    
    # Calculate 1d RSI(14) for momentum confirmation
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_14_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Williams %R signals with RSI confirmation for mean reversion
        # Long: Oversold Williams %R (< -80) + RSI showing weakness (< 30) for reversal
        # Short: Overbought Williams %R (> -20) + RSI showing strength (> 70) for reversal
        if vol_filter:
            if williams_r_14_aligned[i] < -80 and rsi_14_aligned[i] < 30:
                signals[i] = 0.25  # Long mean reversion
            elif williams_r_14_aligned[i] > -20 and rsi_14_aligned[i] > 70:
                signals[i] = -0.25  # Short mean reversion
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_RSI14_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0