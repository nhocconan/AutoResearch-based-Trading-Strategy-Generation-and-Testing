#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Trend_Pullback_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Trend filter (EMA200) ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        ema_val = ema200_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + pullback to EMA + volume confirmation
            if (close_val > ema_val and          # Price above 1d EMA200 (uptrend)
                close_val <= ema_val * 1.02 and  # Within 2% above EMA (pullback zone)
                vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + bounce to EMA + volume confirmation
            elif (close_val < ema_val and        # Price below 1d EMA200 (downtrend)
                  close_val >= ema_val * 0.98 and # Within 2% below EMA (bounce zone)
                  vol_ratio_val > 1.5):          # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or loss of momentum
            if (close_val < ema_val * 0.98 or    # Price falls below bounce zone
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or loss of momentum
            if (close_val > ema_val * 1.02 or    # Price rises above pullback zone
                vol_ratio_val < 0.8):            # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals