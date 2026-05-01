#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend absence when lines intertwine.
# Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish).
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters out weak breakouts.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (Jaw=13, Teeth=8, Lips=5)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Apply forward shifts (using NaN for shifted periods)
    jaw_aligned = np.full_like(jaw, np.nan)
    teeth_aligned = np.full_like(teeth, np.nan)
    lips_aligned = np.full_like(lips, np.nan)
    
    if len(jaw) > jaw_shift:
        jaw_aligned[jaw_shift:] = jaw[:-jaw_shift]
    if len(teeth) > teeth_shift:
        teeth_aligned[teeth_shift:] = teeth[:-teeth_shift]
    if len(lips) > lips_shift:
        lips_aligned[lips_shift:] = lips[:-lips_shift]
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(jaw_shift, teeth_shift, lips_shift, 20) + max(jaw_period, teeth_period, lips_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Williams Alligator signals
        # Bullish: Lips > Teeth > Jaw (lines diverging upward)
        bullish_align = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish: Lips < Teeth < Jaw (lines diverging downward)
        bearish_align = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment, volume spike, uptrend
            if bullish_align and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment, volume spike, downtrend
            elif bearish_align and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish Alligator alignment or trend reversal
            if bearish_align or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish Alligator alignment or trend reversal
            if bullish_align or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals