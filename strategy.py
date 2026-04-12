#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1w_vix_signal_v1
# Uses weekly VIX-like volatility index (based on weekly ATR) to filter regime.
# Long when price breaks above weekly VWAP with expanding volatility (weekly ATR > 1.5x 10-week avg).
# Short when price breaks below weekly VWAP with expanding volatility.
# Exits when volatility contracts (weekly ATR < 0.8x 10-week avg) or price crosses VWAP in opposite direction.
# Designed for low trade frequency (target: 20-30 trades/year) with volatility expansion bias.
# Works in trending markets via volatility breakouts and avoids ranging markets via volatility contraction filter.

name = "4h_1w_vix_signal_v1"
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
    
    # Get weekly data for VIX-like volatility and VWAP
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR (True Range based)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for weekly
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(10) weekly
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Weekly VWAP approximation: (high + low + close)/3 as typical price
    typical_price = (high_1w + low_1w + close_1w) / 3.0
    # Volume-weighted average price approximation using typical price and volume
    vwap_num = pd.Series(typical_price * df_1w['volume'].values).rolling(window=10, min_periods=10).sum().values
    vwap_den = pd.Series(df_1w['volume'].values).rolling(window=10, min_periods=10).sum().values
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Align weekly data to 4h timeframe
    atr_10_aligned = align_htf_to_ltf(prices, df_1w, atr_10)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if np.isnan(atr_10_aligned[i]) or np.isnan(vwap_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volatility expansion/contraction
        vol_expanding = atr_10_aligned[i] > (np.nanmedian(atr_10_aligned[max(0, i-50):i+1]) * 1.5)
        vol_contracting = atr_10_aligned[i] < (np.nanmedian(atr_10_aligned[max(0, i-50):i+1]) * 0.8)
        
        # Exit conditions: volatility contraction or VWAP crossover in opposite direction
        if position == 1 and (vol_contracting or close[i] < vwap_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (vol_contracting or close[i] > vwap_aligned[i]):
            position = 0
            signals[i] = 0.0
        # Entry conditions: volatility expansion + price breaks VWAP
        elif vol_expanding and vol_confirm[i]:
            if close[i] > vwap_aligned[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif close[i] < vwap_aligned[i] and position != -1:
                position = -1
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals