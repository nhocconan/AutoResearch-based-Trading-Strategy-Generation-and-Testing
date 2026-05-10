#!/usr/bin/env python3
# 1h_PSAR_Trend_Filter_4hVolatility
# Hypothesis: Parabolic SAR on 1h captures medium-term trends, filtered by 4h volatility regime (ATR-based).
# Low volatility periods indicate consolidation; high volatility indicates trending markets.
# Only trade when PSAR signals align with high volatility regime to avoid whipsaws in ranging markets.
# Designed for 1h timeframe with moderate trade frequency (target: 15-35 trades/year) suitable for both bull and bear markets.

name = "1h_PSAR_Trend_Filter_4hVolatility"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_psar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Calculate Parabolic SAR values."""
    n = len(high)
    psar = np.zeros(n)
    psar_bull = np.zeros(n)  # SAR during uptrend
    psar_bear = np.zeros(n)  # SAR during downtrend
    bull = np.ones(n, dtype=bool)  # True for uptrend
    
    # Initialize
    psar[0] = low[0]
    psar_bull[0] = low[0]
    psar_bear[0] = high[0]
    af = af_start
    ep = high[0]  # extreme point
    
    for i in range(1, n):
        if bull[i-1]:  # was uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Check for reversal
            if psar[i] > low[i]:
                bull[i] = False  # reverse to downtrend
                psar[i] = psar_bear[i-1]  # SAR becomes previous bear SAR
                af = af_start
                ep = low[i]
            else:
                bull[i] = True
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_increment, af_max)
                psar_bull[i] = psar[i]
                psar_bear[i] = psar_bear[i-1]
        else:  # was downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Check for reversal
            if psar[i] < high[i]:
                bull[i] = True  # reverse to uptrend
                psar[i] = psar_bull[i-1]  # SAR becomes previous bull SAR
                af = af_start
                ep = high[i]
            else:
                bull[i] = False
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_increment, af_max)
                psar_bear[i] = psar[i]
                psar_bull[i] = psar_bull[i-1]
    
    return psar

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for volatility filter (ATR-based)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) on 4h for volatility regime
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = np.zeros_like(close_4h)
    atr_4h[:14] = np.nan
    for i in range(14, len(close_4h)):
        atr_4h[i] = np.mean(tr[i-13:i+1])
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma_50 = np.zeros_like(atr_4h)
    atr_ma_50[:50] = np.nan
    for i in range(50, len(atr_4h)):
        if not np.isnan(atr_4h[i-49:i+1]).any():
            atr_ma_50[i] = np.mean(atr_4h[i-49:i+1])
    
    vol_ratio = np.zeros_like(atr_4h)
    vol_ratio[:] = np.nan
    for i in range(len(atr_4h)):
        if not np.isnan(atr_4h[i]) and not np.isnan(atr_ma_50[i]) and atr_ma_50[i] > 0:
            vol_ratio[i] = atr_4h[i] / atr_ma_50[i]
    
    # Align 4h volatility ratio to 1h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio)
    
    # Calculate PSAR on 1h price data
    psar = calculate_psar(high, low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for volatility ratio
    
    for i in range(start_idx, n):
        if np.isnan(vol_ratio_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in high volatility regime (vol_ratio > 1.2 indicates trending market)
        high_volatility = vol_ratio_aligned[i] > 1.2
        
        if position == 0:
            # Long: price above PSAR and high volatility
            if close[i] > psar[i] and high_volatility:
                signals[i] = 0.20
                position = 1
            # Short: price below PSAR and high volatility
            elif close[i] < psar[i] and high_volatility:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long: maintain if price above PSAR and high volatility, else exit
            if close[i] > psar[i] and high_volatility:
                signals[i] = 0.20
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: maintain if price below PSAR and high volatility, else exit
            if close[i] < psar[i] and high_volatility:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
                position = 0
    
    return signals