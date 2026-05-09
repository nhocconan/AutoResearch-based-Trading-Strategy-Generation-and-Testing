#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ChaikinOscillator_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Chaikin Oscillator (60-period) on 6h data
    # Chaikin Money Flow = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    # Chaikin Oscillator = EMA3(CMF) - EMA10(CMF)
    
    # Calculate Money Flow Multiplier
    mfm = np.zeros_like(close)
    denominator = (high - low)
    # Avoid division by zero
    mask = denominator != 0
    mfm[mask] = ((close[mask] - low[mask]) - (high[mask] - close[mask])) / denominator[mask]
    mfm[~mask] = 0.0
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Calculate Chaikin Oscillator: EMA3(MFV) - EMA10(MFV)
    mfv_series = pd.Series(mfv)
    ema3 = mfv_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = mfv_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3 - ema10
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 10)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(chaikin_osc[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        chaikin = chaikin_osc[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Chaikin Oscillator crosses above 0 + above 12h EMA50 + volume spike
            if chaikin > 0 and chaikin_osc[i-1] <= 0 and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Chaikin Oscillator crosses below 0 + below 12h EMA50 + volume spike
            elif chaikin < 0 and chaikin_osc[i-1] >= 0 and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Chaikin Oscillator crosses below 0 or below 12h EMA50
            if chaikin < 0 and chaikin_osc[i-1] >= 0 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Chaikin Oscillator crosses above 0 or above 12h EMA50
            if chaikin > 0 and chaikin_osc[i-1] <= 0 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals