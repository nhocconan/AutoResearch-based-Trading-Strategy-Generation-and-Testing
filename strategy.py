#!/usr/bin/env python3
"""
6h_HTF_1d_CCI_Reversal_V1
Hypothesis: 6h CCI extreme reversals with 1d EMA200 trend filter. CCI > +100 indicates overbought (short signal when price < EMA200), CCI < -100 indicates oversold (long signal when price > EMA200). Uses 6h primary timeframe with 1d HTF for trend filter. Targets mean reversion in ranging markets and pullbacks in trending markets, works in both bull and bear regimes by aligning with higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA200 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d EMA200 for trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # CCI(20) calculation
    typical_price = (high_6h + low_6h + close_6h) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    tp_mad = np.where(tp_mad == 0, 1e-10, tp_mad)
    cci = (typical_price - tp_ma) / (0.015 * tp_mad)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(cci[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Long: CCI oversold (< -100) and price above 1d EMA200 (uptrend alignment)
            if cci[i] < -100 and price > ema_200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI overbought (> +100) and price below 1d EMA200 (downtrend alignment)
            elif cci[i] > 100 and price < ema_200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CCI returns to neutral (> -50) or trend breaks
            if cci[i] > -50 or price < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CCI returns to neutral (< +50) or trend breaks
            if cci[i] < 50 or price > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_CCI_Reversal_V1"
timeframe = "6h"
leverage = 1.0