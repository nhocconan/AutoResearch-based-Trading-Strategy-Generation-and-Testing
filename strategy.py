#!/usr/bin/env python3
"""
1D Keltner Channel Breakout with Volume Spike and Volatility Regime Filter
Long when price closes above upper Keltner band with volume > 2x 20-day average
and ATR(14) < 1.5x its 50-day average (low volatility regime).
Short when price closes below lower Keltner band with volume spike and low volatility.
Exit when price returns to middle band or volatility spikes (ATR > 2x 50-day avg).
Designed for 1d timeframe to capture breakouts from low volatility regimes in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for volatility regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ATR(14) for Keltner channels and volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period EMA for Keltner middle band
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: ±2 * ATR around EMA20
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # Weekly ATR for volatility regime filter
    tr1_w = df_1w['high'].values[1:] - df_1w['low'].values[1:]
    tr2_w = np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1])
    tr3_w = np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    atr_w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 50-week ATR average for volatility regime
    atr_w_ma50 = pd.Series(atr_w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly ATR and its MA to daily timeframe
    atr_w_aligned = align_htf_to_ltf(prices, df_1w, atr_w)
    atr_w_ma50_aligned = align_htf_to_ltf(prices, df_1w, atr_w_ma50)
    
    # Volume confirmation: 20-day volume average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema20[i]) or np.isnan(atr_w_aligned[i]) or 
            np.isnan(atr_w_ma50_aligned[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        atr_w_now = atr_w_aligned[i]
        atr_w_ma = atr_w_ma50_aligned[i]
        
        # Low volatility regime: weekly ATR < 1.5x its 50-day average
        low_vol = atr_w_now < 1.5 * atr_w_ma
        # High volatility exit: weekly ATR > 2x its 50-day average
        high_vol = atr_w_now > 2.0 * atr_w_ma
        
        if position == 0:
            # Long: close above upper Keltner band with volume spike in low vol regime
            if price > keltner_upper[i] and vol > 2.0 * vol_ma and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: close below lower Keltner band with volume spike in low vol regime
            elif price < keltner_lower[i] and vol > 2.0 * vol_ma and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or volatility spikes
            if price < ema20[i] or high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or volatility spikes
            if price > ema20[i] or high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_Keltner_Breakout_Volume_Volatility"
timeframe = "1d"
leverage = 1.0