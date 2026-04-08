#!/usr/bin/env python3
# 12h_1d_vwap_deviation_mean_reversion_v1
# Hypothesis: Mean reversion from daily VWAP on 12h timeframe. Long when price deviates significantly below daily VWAP,
# short when price deviates significantly above daily VWAP. Uses 1d volatility filter to avoid low-volatility chop.
# Designed for 12-30 trades/year on 12h to avoid fee drag. Works in bull/bear via mean reversion logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vwap_deviation_mean_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price for 1d
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Calculate cumulative volume and cumulative volume * typical price for VWAP
    cum_vol_1d = np.cumsum(volume_1d)
    cum_vol_price_1d = np.cumsum(volume_1d * typical_price_1d)
    
    # Calculate VWAP (avoid division by zero)
    vwap_1d = np.divide(cum_vol_price_1d, cum_vol_1d, out=np.full_like(cum_vol_price_1d, np.nan), where=cum_vol_1d!=0)
    
    # Align 1d VWAP to 12h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 1d ATR for volatility filter (14 periods)
    # True Range components
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    # Insert first value (no previous close)
    tr_1d = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr_1d])
    
    # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_1d = np.full_like(tr_1d, np.nan)
    if len(tr_1d) >= 14:
        # First ATR is simple average of first 14 TR values
        atr_1d[13] = np.mean(tr_1d[0:14])
        # Subsequent values: ATR = (prev_atr * 13 + current_tr) / 14
        for i in range(14, len(tr_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Align 1d ATR to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 14  # Ensure ATR is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Calculate deviation from VWAP as percentage of ATR
        deviation = (close[i] - vwap_1d_aligned[i]) / atr_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP or opposite deviation
            if deviation >= -0.5:  # Returned halfway to VWAP
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to VWAP or opposite deviation
            if deviation <= 0.5:  # Returned halfway to VWAP
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price significantly below VWAP (-2.0 ATR deviation) with sufficient volatility
            if deviation <= -2.0 and atr_1d_aligned[i] > 0:
                position = 1
                signals[i] = 0.25
            # Short entry: price significantly above VWAP (+2.0 ATR deviation) with sufficient volatility
            elif deviation >= 2.0 and atr_1d_aligned[i] > 0:
                position = -1
                signals[i] = -0.25
    
    return signals