# 12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1 (Hypothesis)
# Strategy: Camarilla pivot breakout with volume confirmation and ATR stop.
# Timeframe: 12h, HTF: 1d. Works in bull/bear via mean-reversion at R1/S1 levels.
# Targets 15-30 trades/year to avoid fee drag. Uses discrete sizing (0.25).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pp = (high_1d + low_1d + close_1d) / 3
    # R1 and S1
    r1 = pp + (high_1d - low_1d) * 1.0833
    s1 = pp - (high_1d - low_1d) * 1.0833
    
    # Align to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h: Price, volume, ATR ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(14) for stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after ATR warmup
        # Skip if any value is NaN
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: Close > R1 + volume confirmation
            if (close_val > r1_val and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Close < S1 + volume confirmation
            elif (close_val < s1_val and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: stop loss or mean reversion
            if (close_val < entry_price - 1.5 * atr_val or  # ATR stop
                close_val < s1_val):                      # Mean reversion to S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or mean reversion
            if (close_val > entry_price + 1.5 * atr_val or  # ATR stop
                close_val > r1_val):                        # Mean reversion to R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals