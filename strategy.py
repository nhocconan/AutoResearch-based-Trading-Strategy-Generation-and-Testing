#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price action combined with daily momentum and volume confirmation
# Strategy uses daily RSI(14) for momentum filter (RSI > 50 for long, < 50 for short)
# Enter on 4h close crossing above/below daily VWAP with volume spike
# Exit on opposite signal or trend reversal
# Designed for fewer trades (~25-40/year) with clear trend alignment to work in bull/bear markets

name = "4h_DailyVWAP_RSI_Momentum_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily VWAP (typical price * volume) / cumulative volume
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (tp_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Calculate daily RSI(14)
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align daily indicators to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike: current volume > 1.8 * 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vwap_val = vwap_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: close above VWAP + RSI > 50 + volume spike
            if (close[i] > vwap_val and 
                rsi_val > 50 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: close below VWAP + RSI < 50 + volume spike
            elif (close[i] < vwap_val and 
                  rsi_val < 50 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below VWAP OR RSI < 40 (momentum fade)
            if (close[i] < vwap_val or rsi_val < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above VWAP OR RSI > 60 (momentum fade)
            if (close[i] > vwap_val or rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals