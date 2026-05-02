#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide intraday support/resistance structure
# R3/S3 levels represent strong reversal/breakout zones
# 1d EMA34 filters trades to align with higher timeframe trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Targets 50-150 trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Need previous day's high, low, close
    df_1d_prev = df_1d.shift(1)
    high_1d_prev = df_1d_prev['high'].values
    low_1d_prev = df_1d_prev['low'].values
    close_1d_prev = df_1d_prev['close'].values
    
    # Align previous day's OHLC to 6h timeframe
    high_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, high_1d_prev.values)
    low_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, low_1d_prev.values)
    close_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, close_1d_prev.values)
    
    # Calculate Camarilla levels
    range_1d = high_1d_prev_aligned - low_1d_prev_aligned
    camarilla_h5 = close_1d_prev_aligned + range_1d * 1.1/2  # R3
    camarilla_h4 = close_1d_prev_aligned + range_1d * 1.1/4  # R2
    camarilla_h3 = close_1d_prev_aligned + range_1d * 1.1/6  # R1
    camarilla_l3 = close_1d_prev_aligned - range_1d * 1.1/6  # S1
    camarilla_l2 = close_1d_prev_aligned - range_1d * 1.1/4  # S2
    camarilla_l1 = close_1d_prev_aligned - range_1d * 1.1/2  # S3
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 (camarilla_h3) + price > 1d EMA34 + volume spike
            if close[i] > camarilla_h3[i] and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 (camarilla_l3) + price < 1d EMA34 + volume spike
            elif close[i] < camarilla_l3[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 (camarilla_l3)
            if close[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 (camarilla_h3)
            if close[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals