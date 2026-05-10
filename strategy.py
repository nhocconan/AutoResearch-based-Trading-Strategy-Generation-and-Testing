# 4h_Camarilla_Pivot_R1_S1_Breakout_1dATR_Trend_Filter
# Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) as key support/resistance levels.
# Enters long when price breaks above R1 with 1d ATR filter and volume confirmation.
# Enters short when price breaks below S1 with 1d ATR filter and volume confirmation.
# Exits when price returns to the pivot point (PP) or reverses across the opposite level.
# Uses 1d ATR to filter out low volatility periods and avoid false breakouts.
# Designed for 30-50 trades/year on 4h to balance opportunity with fee efficiency.

name = "4h_Camarilla_Pivot_R1_S1_Breakout_1dATR_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R1, S1, PP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Calculate 1d ATR(14) for volatility filter
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.zeros(len(df_1d))
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[1:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    else:
        # Not enough data for ATR, use zeros
        pass
    
    # Align 1d indicators to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume moving average (20-period) for confirmation
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ATR filter: only trade when volatility is sufficient (ATR > 0)
        atr_filter = atr_1d_aligned[i] > 0
        
        if position == 0:
            # Long: Price breaks above R1 with volume and ATR confirmation
            if (close[i] > r1_1d_aligned[i] and 
                vol_confirm and atr_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and ATR confirmation
            elif (close[i] < s1_1d_aligned[i] and 
                  vol_confirm and atr_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to PP or breaks below S1 (reversal)
            if (close[i] <= pp_1d_aligned[i] or 
                close[i] < s1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to PP or breaks above R1 (reversal)
            if (close[i] >= pp_1d_aligned[i] or 
                close[i] > r1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals