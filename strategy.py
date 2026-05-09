#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_Trend_Confirm_With_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate TRIX (15-period) on daily close
    close_1d = df_1d['close'].values
    # EMA1
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix_raw[0] = 0
    
    # Align TRIX to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # Get 4h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detection (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_4h[i]) or np.isnan(ema20[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: TRIX > 0 (bullish momentum) + price above EMA20 + volume spike
            if trix_4h[i] > 0 and close[i] > ema20[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX < 0 (bearish momentum) + price below EMA20 + volume spike
            elif trix_4h[i] < 0 and close[i] < ema20[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns negative OR price falls below EMA20
            if trix_4h[i] < 0 or close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive OR price rises above EMA20
            if trix_4h[i] > 0 or close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals