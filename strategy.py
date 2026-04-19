#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 100-day EMA trend filter + 1d ATR volatility filter + volume confirmation.
# Uses long-term EMA (100d) to filter trend direction, with ATR-based volatility expansion
# to confirm breakouts. Volume confirms institutional participation. Works in bull/bear
# by following strong trends and avoiding chop. Target: 15-25 trades/year per symbol.
name = "12h_EMA100_ATRVol_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA100 trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA100 on daily (100-day exponential moving average)
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Calculate ATR (14-period) on daily
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 120  # Ensure EMA100 and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_100_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_100_val = ema_100_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volatility expansion filter: current ATR > 1.2x average ATR
        # This ensures we only trade during volatile, trending periods
        atr_expanded = atr_val > 1.2 * np.nanmedian(atr_14_aligned[max(0, i-50):i])
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Enter long if price above EMA100, volatility expanding, and volume confirmation
            if price > ema_100_val and atr_expanded and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price below EMA100, volatility expanding, and volume confirmation
            elif price < ema_100_val and atr_expanded and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below EMA100 or volatility contracts
            if price < ema_100_val or not atr_expanded:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above EMA100 or volatility contracts
            if price > ema_100_val or not atr_expanded:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals