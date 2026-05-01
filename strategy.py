#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend absence (all lines intertwined) 
# and trend emergence (lines diverge). Long when Lips > Teeth > Jaw with volume spike and 1w uptrend.
# Short when Lips < Teeth < Jaw with volume spike and 1w downtrend.
# Designed for low-frequency, high-conviction trades to minimize fee drag.
# Target: 15-25 trades/year for sustainable performance in both bull and bear markets.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (need enough for SMMA and shifts)
    start_idx = 50  # Need sufficient history for EMA50 and SMMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA50 direction
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Alligator conditions
        lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
        teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
        lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
        teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips > Teeth > Jaw (alligator waking up bullish), volume spike, 1w uptrend
            if lips_above_teeth and teeth_above_jaw and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (alligator waking up bearish), volume spike, 1w downtrend
            elif lips_below_teeth and teeth_below_jaw and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when alligator lines intertwine (trend losing strength) or trend reversal
            if not (lips_above_teeth and teeth_above_jaw) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when alligator lines intertwine (trend losing strength) or trend reversal
            if not (lips_below_teeth and teeth_below_jaw) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals