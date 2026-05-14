# 4h_20241215_Performance_Enhancement_1
# Hypothesis: Enhance the proven Camarilla R1/S1 breakout strategy with volume confirmation and trend filter by adding a volatility-based regime filter (ATR-based) to reduce false breakouts during low volatility periods. This should improve the win rate and reduce whipsaws in both bull and bear markets while maintaining reasonable trade frequency.
# The strategy uses 4h timeframe with 1d HTF for Camarilla levels and EMA trend, plus volume spike and volatility regime filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_20241215_Performance_Enhancement_1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily high/low/close for Camarilla pivot (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d_prev + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d_prev - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for volatility regime filter (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: ATR > 20-period average ATR (avoid low volatility whipsaws)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_regime = atr > atr_ma  # High volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        vol_reg = vol_regime[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + uptrend + volume spike + high volatility regime
            if (close[i] > r1 and 
                close[i] > ema34_1d_val and 
                vol_spike and 
                vol_reg):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + downtrend + volume spike + high volatility regime
            elif (close[i] < s1 and 
                  close[i] < ema34_1d_val and 
                  vol_spike and 
                  vol_reg):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR trend turns down
            if (close[i] < s1 or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 OR trend turns up
            if (close[i] > r1 or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals