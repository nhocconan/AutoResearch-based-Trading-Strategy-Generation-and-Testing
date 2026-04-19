#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d momentum (RSI) + volume confirmation + volatility filter.
# Uses daily RSI for momentum direction, requires volume above average and low volatility.
# Aims for 50-150 total trades over 4 years by filtering for high-probability setups.
# Works in bull/bear via RSI extremes and volatility filter to avoid chop.
name = "12h_1d_RSI_Momentum_Volume_Volatility"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and ATR calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate RSI on 1d timeframe (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate ATR on 1d timeframe for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: volume > 1.2 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.2)
    
    # Volatility filter: ATR < 1.5 * 50-period average (avoid high volatility chop)
    atr_ma = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_1d_aligned < (atr_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr_ma[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        rsi = rsi_1d_aligned[i]
        vol_ok = volume_filter[i]
        vol_filter_ok = volatility_filter[i]
        
        if position == 0:
            # Long when RSI > 55 (bullish momentum) with volume and low volatility
            if rsi > 55 and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short when RSI < 45 (bearish momentum) with volume and low volatility
            elif rsi < 45 and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when RSI < 40 (momentum fading)
            if rsi < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when RSI > 60 (momentum fading)
            if rsi > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals