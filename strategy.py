#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12/26 EMA crossover with 1d VWAP filter and volume confirmation
# Long when fast EMA crosses above slow EMA, price > 1d VWAP, and volume spike
# Short when fast EMA crosses below slow EMA, price < 1d VWAP, and volume spike
# Uses 1d VWAP as trend filter to ensure alignment with daily trend
# Volume confirmation reduces false breakouts
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)
name = "4h_EMA12_26_1dVWAP_Volume"
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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (cumulative)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    
    # Align 1d VWAP to 4h timeframe (wait for daily close)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate EMA12 and EMA26 on 4h close
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)  # Need EMA26 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema12[i]) or 
            np.isnan(ema26[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        ema_fast = ema12[i]
        ema_slow = ema26[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        # EMA crossover signals
        ema_cross_up = ema_fast > ema_slow and ema12[i-1] <= ema26[i-1]
        ema_cross_down = ema_fast < ema_slow and ema12[i-1] >= ema26[i-1]
        
        if position == 0:
            # Enter long: EMA bullish crossover, price > VWAP, volume confirmed
            if ema_cross_up and price > vwap and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: EMA bearish crossover, price < VWAP, volume confirmed
            elif ema_cross_down and price < vwap and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when EMA bearish crossover or price < VWAP
            if ema_cross_down or price < vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when EMA bullish crossover or price > VWAP
            if ema_cross_up or price > vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals