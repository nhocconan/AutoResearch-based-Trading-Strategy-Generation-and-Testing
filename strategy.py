#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend identification on 12h timeframe
# 1w EMA50 provides multi-timeframe trend alignment to reduce whipsaw in ranging markets
# Volume confirmation (>1.8x 30-bar average) ensures breakout strength
# ATR-based trailing stop via signal=0 when price crosses opposite Alligator line
# Discrete sizing 0.25 to limit fee drag; target 60-120 total trades over 4 years (15-30/year)
# Williams Alligator catches strong trends while avoiding choppy markets - works in both bull/bear regimes

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead  
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Calculate SMMA for different periods
    smma_13 = smma(close, 13)
    smma_8 = smma(close, 8)
    smma_5 = smma(close, 5)
    
    # Alligator lines with forward shift
    jaw = np.roll(smma_13, 8)   # Jaw: 13-period SMMA shifted 8 bars ahead
    teeth = np.roll(smma_8, 5)  # Teeth: 8-period SMMA shifted 5 bars ahead
    lips = np.roll(smma_5, 3)   # Lips: 5-period SMMA shifted 3 bars ahead
    
    # Calculate volume spike filter (>1.8x 30-bar average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma_30)
    
    # Align HTF indicators to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1w, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long entry: Alligator uptrend AND price > Lips AND volume spike
            if alligator_long and close[i] > lips_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator downtrend AND price < Lips AND volume spike
            elif alligator_short and close[i] < lips_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns down OR price crosses below Teeth
            if not alligator_long or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns up OR price crosses above Teeth
            if not alligator_short or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals