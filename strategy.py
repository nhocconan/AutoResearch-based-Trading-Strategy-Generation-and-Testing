#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-day trend filter and volume confirmation
# Long when Alligator Lips cross above Teeth (bullish alignment) + daily EMA(50) uptrend + volume spike
# Short when Alligator Lips cross below Teeth (bearish alignment) + daily EMA(50) downtrend + volume spike
# Alligator uses SMAs (13,8,5) with future shifts (8,5,3) - effective in trending and ranging markets
# Daily trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Alligator components (12h timeframe)
    # Jaw (blue): 13-period SMMA shifted 8 bars ahead
    # Teeth (red): 8-period SMMA shifted 5 bars ahead  
    # Lips (green): 5-period SMMA shifted 3 bars ahead
    def smoothed_ma(data, period):
        # Smoothed Moving Average (SMMA) - similar to RMA/Wilder's smoothing
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma = np.full_like(sma, np.nan, dtype=float)
        if len(sma) >= period:
            smma[period-1] = sma[period-1]
            for i in range(period, len(sma)):
                smma[i] = (smma[i-1] * (period-1) + sma[i]) / period
        return smma
    
    jaw = smoothed_ma(close, 13)
    teeth = smoothed_ma(close, 8)
    lips = smoothed_ma(close, 5)
    
    # Apply Alligator shifts (future shifts - need to handle carefully to avoid look-ahead)
    # According to Williams: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    # For backward-looking calculation, we use the values as-is and interpret crossovers
    jaw_val = jaw
    teeth_val = teeth
    lips_val = lips
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw_val[i]) or np.isnan(teeth_val[i]) or 
            np.isnan(lips_val[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        jaw_i = jaw_val[i]
        teeth_i = teeth_val[i]
        lips_i = lips_val[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Lips above Teeth (bullish alignment) + daily uptrend + volume spike
            if lips_i > teeth_i and close[i] > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips below Teeth (bearish alignment) + daily downtrend + volume spike
            elif lips_i < teeth_i and close[i] < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips cross below Teeth OR daily trend turns down
            if lips_i <= teeth_i or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips cross above Teeth OR daily trend turns up
            if lips_i >= teeth_i or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals