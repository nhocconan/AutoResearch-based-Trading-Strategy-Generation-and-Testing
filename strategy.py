#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Vortex Indicator (VI) for trend direction and 1-day ATR for volatility filtering.
# Enters long when VI+ crosses above VI- with price above 1-day EMA and volume spike, short when VI- crosses above VI+ with price below 1-day EMA and volume spike.
# Exits when VI reverses or price crosses 1-day EMA. Designed to capture trends in both bull and bear markets by using VI's trend strength.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_Vortex_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex Indicator and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Vortex Indicator components
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = np.abs(high_1d[0] - low_1d[0])
    vm_minus[0] = np.abs(low_1d[0] - high_1d[0])
    
    # Sum over 14 periods
    n1 = len(high_1d)
    vi_plus = np.zeros(n1)
    vi_minus = np.zeros(n1)
    for i in range(14, n1):
        vi_plus[i] = np.sum(vm_plus[i-13:i+1]) / np.sum(tr[i-13:i+1])
        vi_minus[i] = np.sum(vm_minus[i-13:i+1]) / np.sum(tr[i-13:i+1])
    
    # Calculate 1-day EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for EMA and VI
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        ema_val = ema_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: VI+ crosses above VI- + price above EMA + volume spike
            if vi_plus_val > vi_minus_val and vi_plus_aligned[i-1] <= vi_minus_aligned[i-1] and close[i] > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- crosses above VI+ + price below EMA + volume spike
            elif vi_minus_val > vi_plus_val and vi_minus_aligned[i-1] <= vi_plus_aligned[i-1] and close[i] < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- crosses above VI+ or price crosses below EMA
            if vi_minus_val > vi_plus_val or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ crosses above VI- or price crosses above EMA
            if vi_plus_val > vi_minus_val or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals