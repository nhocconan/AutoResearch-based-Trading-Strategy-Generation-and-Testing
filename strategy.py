#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Trend Filter + Volume Spike
# Long when Alligator bullish (jaw < teeth < lips), price > lips, 1d uptrend, volume > 1.5x average
# Short when Alligator bearish (jaw > teeth > lips), price < lips, 1d downtrend, volume > 1.5x average
# Williams Alligator identifies trend direction and alignment; volume confirms strength
# 1d trend filter ensures alignment with higher timeframe momentum
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "4h_WilliamsAlligator_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Williams Alligator (13,8,5) - Smoothed Moving Average (SMMA)
    # Jaw: SMMA(13, 8)
    # Teeth: SMMA(8, 5)
    # Lips: SMMA(5, 3)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: Alligator bullish (jaw < teeth < lips), price > lips, 1d uptrend, volume spike
            if jaw_val < teeth_val and teeth_val < lips_val and close_val > lips_val and ema50_1d_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish (jaw > teeth > lips), price < lips, 1d downtrend, volume spike
            elif jaw_val > teeth_val and teeth_val > lips_val and close_val < lips_val and ema50_1d_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish or price < teeth or 1d trend down
            if jaw_val > teeth_val or close_val < teeth_val or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish or price > teeth or 1d trend up
            if jaw_val < teeth_val or close_val > teeth_val or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals