#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RSI_Momentum_BullBearPower"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 days for calculations
        return np.zeros(n)
    
    # === 1d: Calculate Elder Ray indicators (Bull Power / Bear Power) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA of close (standard for Elder Ray)
    close_series = pd.Series(close_1d)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 6h: RSI(14) for momentum ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to RMA)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h: Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        rsi_val = rsi[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) AND RSI > 50 (uptrend momentum) with volume
            if (bull_val > 0 and 
                rsi_val > 50 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum) AND RSI < 50 (downtrend momentum) with volume
            elif (bear_val < 0 and 
                  rsi_val < 50 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR RSI drops below 50 (momentum fading)
            if (bull_val <= 0 or rsi_val <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive OR RSI rises above 50 (momentum fading)
            if (bear_val >= 0 or rsi_val >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals