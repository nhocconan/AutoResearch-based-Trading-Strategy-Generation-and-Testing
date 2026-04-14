#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-day Parabolic SAR for trend direction and 1-day ATR for volatility filtering.
Long when price > Parabolic SAR and ATR(14) > 20-period SMA of ATR (rising volatility).
Short when price < Parabolic SAR and ATR(14) > 20-period SMA of ATR.
Exit when price crosses Parabolic SAR in opposite direction.
Designed for low turnover: ~20-40 trades/year per symbol to minimize fee drift.
Works in bull via trend following and in bear via short-side symmetry.
Uses volatility expansion to capture breakouts and avoid choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
    n = len(high)
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    af = np.zeros(n)
    ep = np.zeros(n)
    
    sar[0] = low[0]
    trend[0] = 1
    af[0] = af_start
    ep[0] = high[0]
    
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # Prevent SAR from penetrating previous lows
            if i >= 2:
                sar[i] = min(sar[i], low[i-1], low[i-2])
            
            # Trend reversal
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = low[i]
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # Prevent SAR from penetrating previous highs
            if i >= 2:
                sar[i] = max(sar[i], high[i-1], high[i-2])
            
            # Trend reversal
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = high[i]
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    return sar

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data once for Parabolic SAR and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day Parabolic SAR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    sar = calculate_parabolic_sar(high_1d, low_1d)
    
    # 1-day ATR (14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period SMA of ATR for volatility filter
    atr_sma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Previous close for TR calculation
    close_1d = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 1-day index
        idx_1d = i // 6  # 6 bars per day (4h timeframe)
        if idx_1d < 20:  # need enough for ATR/SMA
            continue
        
        # Get previous 1-day values to avoid look-ahead
        sar_prev = sar[idx_1d - 1] if idx_1d - 1 < len(sar) else sar[-1]
        atr_prev = atr[idx_1d - 1] if idx_1d - 1 < len(atr) else atr[-1]
        atr_sma_prev = atr_sma[idx_1d - 1] if idx_1d - 1 < len(atr_sma) else atr_sma[-1]
        
        if np.isnan(sar_prev) or np.isnan(atr_prev) or np.isnan(atr_sma_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        sar_arr = np.full(len(df_1d), sar_prev)
        atr_arr = np.full(len(df_1d), atr_prev)
        atr_sma_arr = np.full(len(df_1d), atr_sma_prev)
        sar_4h = align_htf_to_ltf(prices, df_1d, sar_arr)[i]
        atr_4h = align_htf_to_ltf(prices, df_1d, atr_arr)[i]
        atr_sma_4h = align_htf_to_ltf(prices, df_1d, atr_sma_arr)[i]
        
        if position == 0:
            # Long: price > SAR and rising volatility (ATR > SMA of ATR)
            if (close[i] > sar_4h and 
                atr_4h > atr_sma_4h):
                position = 1
                signals[i] = position_size
            # Short: price < SAR and rising volatility
            elif (close[i] < sar_4h and 
                  atr_4h > atr_sma_4h):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price crosses below SAR (trend reversal)
            if close[i] < sar_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price crosses above SAR (trend reversal)
            if close[i] > sar_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1D_ParabolicSAR_ATR_Volatility"
timeframe = "4h"
leverage = 1.0