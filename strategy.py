#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA(34) trend and volume spike
# Uses 6h primary timeframe with 1d HTF for trend alignment and 1w HTF for pivot direction.
# Breakouts at R3 (long) or S3 (short) in direction of 1d EMA(34) with volume confirmation.
# 1w trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in both bull and bear markets by following the 1d trend direction only.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    camarilla_h4 = typical_price + (range_ * 1.1 / 2)  # R3 equivalent
    camarilla_l4 = typical_price - (range_ * 1.1 / 2)  # S3 equivalent
    camarilla_h5 = typical_price + (range_ * 1.1)      # R4 equivalent
    camarilla_l5 = typical_price - (range_ * 1.1)      # S4 equivalent
    
    # Align pivot levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5.values)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5.values)
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 80  # max(34 for EMA, 20 for 1w EMA, 20 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H4 (R3) + above 1d EMA(34) + above 1w EMA(20) + volume spike
            if (close[i] > h4_aligned[i] and close[i] > ema_34_1d_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla L4 (S3) + below 1d EMA(34) + below 1w EMA(20) + volume spike
            elif (close[i] < l4_aligned[i] and close[i] < ema_34_1d_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below Camarilla L4 (mean reversion) or below 1d EMA(34) (trend reversal)
            if close[i] < l4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above Camarilla H4 (mean reversion) or above 1d EMA(34) (trend reversal)
            if close[i] > h4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals