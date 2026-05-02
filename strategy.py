#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1d EMA34 Trend + Volume Spike
# Uses 12h timeframe with Camarilla pivot levels (R3/S3) from prior 1d for structure,
# 1d EMA34 for trend filter, and volume confirmation for momentum.
# Designed for low trade frequency (12-37/year) to minimize fee drag while capturing
# significant breakouts in both bull and bear markets via HTF trend alignment.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d (using complete 1d bar only)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = df_1d['close'].shift(1).values  # Prior day close
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d_shift + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d_shift - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 2.0x 20-period average (~10 days for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 1d EMA34 and prior 1d data)
    start_idx = 34  # 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike AND price > 1d EMA34 (bullish trend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike AND price < 1d EMA34 (bearish trend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 (mean reversion) OR price < 1d EMA34 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 (mean reversion) OR price > 1d EMA34 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals