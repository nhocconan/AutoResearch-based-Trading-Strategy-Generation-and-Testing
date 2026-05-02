#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw: SMA(13,8), Teeth: SMA(8,5), Lips: SMA(5,3)) to identify trends
# 1d EMA50 filter ensures trades align with higher timeframe trend
# Volume spike (2.0x 12-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in bull markets (Alligator bullish alignment) and bear markets (Alligator bearish alignment)

name = "12h_Williams_Alligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h timeframe
    # Jaw: Blue line - 13-period SMA shifted 8 bars
    # Teeth: Red line - 8-period SMA shifted 5 bars  
    # Lips: Green line - 5-period SMA shifted 3 bars
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = close_s.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = close_s.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 12-period average (12*12h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish (Lips > Teeth > Jaw) AND price > 1d EMA50 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish (Lips < Teeth < Jaw) AND price < 1d EMA50 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price below 1d EMA50 (trend change)
            if (lips[i] < teeth[i] or teeth[i] < jaw[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price above 1d EMA50 (trend change)
            if (lips[i] > teeth[i] or teeth[i] > jaw[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals