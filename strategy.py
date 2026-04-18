#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_CCI_Reversal_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly SMA20 for trend (smooth trend filter)
    sma_20_1w = pd.Series(df_1w['close'].values).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Calculate CCI(14) on daily data
    tp = (high + low + close) / 3.0  # Typical Price
    tp_ma = pd.Series(tp).rolling(window=14, min_periods=14).mean().values
    tp_mad = pd.Series(tp).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (tp - tp_ma) / (0.015 * tp_mad)
    # Handle division by zero or very small values
    cci = np.where(tp_mad == 0, 0, cci)
    
    # Volume spike: current volume > 1.5 * 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_20_1w_aligned[i]) or np.isnan(cci[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        sma_20_val = sma_20_1w_aligned[i]
        cci_val = cci[i]
        
        if position == 0:
            # Long: CCI < -100 (oversold) AND price above weekly SMA20 AND volume spike
            if cci_val < -100 and close_val > sma_20_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI > 100 (overbought) AND price below weekly SMA20 AND volume spike
            elif cci_val > 100 and close_val < sma_20_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI > -100 (exit oversold) or CCI > 50 (momentum shift)
            if cci_val > -100 or cci_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI < 100 (exit overbought) or CCI < -50 (momentum shift)
            if cci_val < 100 or cci_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals