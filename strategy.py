#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 12h data to identify trend direction and entry timing
# 1d EMA50 ensures alignment with higher timeframe trend for institutional bias
# Volume spike (1.8x 30-period average) confirms strong participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 12-25 trades/year (50-100 total over 4 years) for 12h timeframe
# Works in bull markets via Jaw-Teeth-Lips alignment up and in bear markets via alignment down

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Load 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_12h = (high_12h + low_12h) / 2  # Using median price as per Alligator tradition
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_12h, 13)
    teeth = smma(median_12h, 8)
    lips = smma(median_12h, 5)
    
    # Align Alligator lines to primary timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate volume spike (1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume MA)
    start_idx = 30  # buffer for 30-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips > Teeth > Jaw (Alligator mouth up) + price > Lips + 1d close > EMA50 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > lips_aligned[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (Alligator mouth down) + price < Lips + 1d close < EMA50 + volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw) or price < Jaw or 1d trend breaks
            if (lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i] or 
                close[i] < jaw_aligned[i] or close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Jaw < Teeth or Teeth < Lips) or price > Jaw or 1d trend breaks
            if (jaw_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < lips_aligned[i] or 
                close[i] > jaw_aligned[i] or close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals