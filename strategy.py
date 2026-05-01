#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w EMA50 trend filter and volume spike confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify
# trend absence (alligator sleeping) vs trend presence (alligator awakening).
# In 6h timeframe, we trade when the alligator is awakening (teeth > lips for long,
# teeth < lips for short) aligned with weekly trend (price > 1w EMA50 for long,
# price < 1w EMA50 for short). Volume confirmation filters weak signals.
# Designed for low-frequency, high-conviction trades to minimize fee drag.
# Target: 12-30 trades/year for sustainable performance on 6h timeframe.

name = "6h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Williams Alligator components on 6h data
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().values
    
    # Apply the Alligator shifts (Jaw shifted 8, Teeth 5, Lips 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for the shifted values that don't have enough history
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current volume > 2.0 * 50-period average volume
    volume_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (volume_ma_50 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (need 50 for EMA50, 13 for Jaw)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Alligator signals: Teeth > Lips = bullish, Teeth < Lips = bearish
        alligator_bullish = teeth_shifted[i] > lips_shifted[i]
        alligator_bearish = teeth_shifted[i] < lips_shifted[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: alligator bullish, volume spike, uptrend
            if alligator_bullish and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: alligator bearish, volume spike, downtrend
            elif alligator_bearish and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on alligator bearish crossover or trend reversal
            if not alligator_bullish or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on alligator bullish crossover or trend reversal
            if not alligator_bearish or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals