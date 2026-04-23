#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
- Long: Lips > Teeth > Jaw (bullish alignment) + price > 1w EMA50 + volume > 1.8x 20-period avg volume
- Short: Lips < Teeth < Jaw (bearish alignment) + price < 1w EMA50 + volume > 1.8x 20-period avg volume
- Exit: ATR trailing stop (2.0x ATR from extreme) OR Alligator alignment breaks
- Uses 1w EMA50 as higher timeframe trend filter to align with weekly momentum
- Volume confirmation reduces false signals in ranging markets
- ATR trailing stop manages risk during strong trends
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag on 1d timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.8x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator: Smoothed Moving Average (SMMA) with shifts
    # Jaw: SMMA(13, 8) - EMA13 of close, shifted 8 bars forward
    # Teeth: SMMA(8, 5) - EMA8 of close, shifted 5 bars forward
    # Lips: SMMA(5, 3) - EMA5 of close, shifted 3 bars forward
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 13, 8, 5, 50)  # Need 20 for volume MA, 14 for ATR, 13 for Jaw, 8 for Teeth, 5 for Lips, 50 for 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]  # Lips > Teeth > Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]  # Lips < Teeth < Jaw
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish alignment + price > 1w EMA50 + volume spike
            if bullish_alignment and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Bearish alignment + price < 1w EMA50 + volume spike
            elif bearish_alignment and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Alligator alignment breaks (not bullish)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            alignment_exit = not bullish_alignment
            
            if trailing_stop_long or alignment_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from short extreme (trailing stop)
            # 2. Alligator alignment breaks (not bearish)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            alignment_exit = not bearish_alignment
            
            if trailing_stop_short or alignment_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0