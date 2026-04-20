#!/usr/bin/env python3
"""
1d_4h_Pivot_R1S1_Breakout_Volume_Control
Daily Camarilla pivot breakout on 4h chart with volume confirmation and RSI filter.
Targets 20-40 trades/year per symbol by requiring:
- Price breaks above R1 (long) or below S1 (short) on 4h close
- Volume > 1.5x 20-period average
- RSI(14) between 40-60 to avoid overextended entries
- Works in bull/bear: pivot levels adapt to volatility, volume filter ensures conviction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_4h_Pivot_R1S1_Breakout_Volume_Control"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla pivot levels (using previous day's OHLC) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = close_1d + range_1d * 1.1 / 12.0
    s1 = close_1d - range_1d * 1.1 / 12.0
    
    # Align to 4h - these levels are fixed for the entire day after daily close
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h: Price, volume, RSI ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after RSI/volume warmup
        # Get values
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi_values[i]
        
        # Skip if any value is NaN
        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_ratio_val) or np.isnan(rsi_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above R1 with volume and RSI not overbought
            if (close_val > r1_val and 
                vol_ratio_val > 1.5 and 
                40 <= rsi_val <= 60):
                signals[i] = 0.25
                position = 1
            # Short: price closes below S1 with volume and RSI not oversold
            elif (close_val < s1_val and 
                  vol_ratio_val > 1.5 and 
                  40 <= rsi_val <= 60):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below pivot or RSI overbought
            if (close_val < pivot_aligned[i] or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above pivot or RSI oversold
            if (close_val > pivot_aligned[i] or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals