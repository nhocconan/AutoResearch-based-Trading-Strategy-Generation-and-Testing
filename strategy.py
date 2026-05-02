#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction via SMAs on median price.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
# 1d EMA50 ensures trades align with daily trend to avoid false signals in chop.
# Volume confirmation (>1.5x 24-period average) filters low-momentum breakouts.
# Designed for 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Works in bull markets (buying bullish Alligator alignment in uptrend) and bear markets
# (selling bearish Alligator alignment in downtrend) by only taking trades in direction of 1d EMA50.

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components (SMAs on median price)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = EMA with alpha = 1/period
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 1.5x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and EMA50)
    start_idx = max(50, 30)  # 50 for EMA50, 30 for prior day data
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND volume spike AND price > 1d EMA50
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) AND volume spike AND price < 1d EMA50
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR price < 1d EMA50
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR price > 1d EMA50
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals