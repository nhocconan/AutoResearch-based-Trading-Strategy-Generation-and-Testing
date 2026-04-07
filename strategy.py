#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Volatility Expansion with Volume Confirmation
# Hypothesis: Combines daily volatility regime detection (ATR ratio) with 4h breakouts and volume confirmation.
# Works in bull via upward breakouts in high volatility regime, in bear via downward breakdowns.
# Uses daily ATR ratio to filter for volatility expansion periods, reducing false breakouts.
# Target: 20-40 trades/year to minimize fee drag.
name = "4h_daily_vol_expansion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility regime and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(10) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                            np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    # Calculate 50-period average ATR for regime
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    # Volatility regime: current ATR > 1.2 * average ATR (expansion phase)
    vol_regime = atr_1d > (1.2 * atr_ma_1d)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4h ATR(12) for breakout threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(12, n):
        # Skip if required data not available
        if (np.isnan(atr_4h[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade during volatility expansion
        if not vol_regime_aligned[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below VWAP approximation (high+low+close)/3 - ATR
            vwap_approx = (high[i] + low[i] + close[i]) / 3
            if close[i] < (vwap_approx - atr_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above VWAP approximation + ATR
            vwap_approx = (high[i] + low[i] + close[i]) / 3
            if close[i] > (vwap_approx + atr_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above VWAP + ATR + volume confirmation + vol regime
            vwap_approx = (high[i] + low[i] + close[i]) / 3
            if (close[i] > (vwap_approx + atr_4h[i]) and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below VWAP - ATR + volume confirmation + vol regime
            elif (close[i] < (vwap_approx - atr_4h[i]) and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals