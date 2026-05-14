# 1d_1w_Pivot_R1S1_Breakout_VolumeATR_v1
# Hypothesis: Weekly Pivot R1/S1 breakouts with volume confirmation and ATR-based trailing stop work across bull/bear cycles. Breakouts indicate institutional interest; volume confirms validity; ATR stop adapts to volatility.
# Timeframe: 1d for balance of signal quality and trade frequency. Weekly pivot provides structural context; daily execution avoids overtrading.
# Expected: 20-50 trades/year per symbol. Works in trends (breakout continuation) and ranges (false breakouts filtered by volume/vol).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Pivot_R1S1_Breakout_VolumeATR_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w high, low, close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w pivot points: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align 1w pivot levels to 1d timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 1w ATR for volatility filter (14-period)
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.absolute(high_1w[1:] - close_1w[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14_1w = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average (1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        atr = atr_14_1w_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below pivot or ATR-based stop
            if price < pivot or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above pivot or ATR-based stop
            if price > pivot or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals