#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly breakout with volume confirmation and volatility filter
# Enter long when price breaks above weekly high AND volume > 2x average AND volatility contraction
# Enter short when price breaks below weekly low AND volume > 2x average AND volatility contraction
# Exit when price returns to weekly midpoint or volatility expands
# Uses weekly structure to capture breakouts in both bull and bear markets with volume confirmation

name = "1d_weekly_breakout_volatility_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for breakout levels
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly high/low aligned to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, high_weekly)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, low_weekly)
    weekly_mid = (weekly_high_aligned + weekly_low_aligned) / 2
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # Volatility filter: ATR(14) contraction (current ATR < 0.8 * 20-period ATR mean)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_contraction = atr < (0.8 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(volume_threshold[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to weekly midpoint OR volatility expansion
            if close[i] <= weekly_mid[i] or not vol_contraction[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to weekly midpoint OR volatility expansion
            if close[i] >= weekly_mid[i] or not vol_contraction[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: price breaks weekly level + volume + volatility contraction
            if volume[i] > volume_threshold[i] and vol_contraction[i]:
                if close[i] > weekly_high_aligned[i]:
                    # Break above weekly high with volume and low volatility
                    signals[i] = 0.25
                    position = 1
                elif close[i] < weekly_low_aligned[i]:
                    # Break below weekly low with volume and low volatility
                    signals[i] = -0.25
                    position = -1
    
    return signals