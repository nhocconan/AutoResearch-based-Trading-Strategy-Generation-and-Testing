#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray combo with 1w trend filter and volume confirmation
# Uses 1d primary timeframe for lower trade frequency (target: 30-100 trades over 4 years)
# Williams Alligator identifies trend direction via smoothed medians (Jaw/Teeth/Lips)
# Elder Ray measures bull/bear power relative to EMA13 for momentum confirmation
# 1w EMA50 ensures alignment with weekly trend, effective in both bull and bear regimes
# Volume spike (1.8x 20-period average) confirms institutional participation
# Designed with tight entry conditions to minimize fee drag while maintaining edge
# Target: 40-80 total trades over 4 years (10-20/year) - within proven winning range for 1d

name = "1d_WilliamsAlligator_ElderRay_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: Smoothed medians (Jaw=13,8; Teeth=8,5; Lips=5,3)
    # Alligator Jaw (blue): 13-period SMMA, shifted 8 bars
    close_series = pd.Series(close)
    smma_13 = close_series.rolling(window=13, min_periods=13).mean()
    alligator_jaw = smma_13.shift(8).values
    
    # Alligator Teeth (red): 8-period SMMA, shifted 5 bars
    smma_8 = close_series.rolling(window=8, min_periods=8).mean()
    alligator_teeth = smma_8.shift(5).values
    
    # Alligator Lips (green): 5-period SMMA, shifted 3 bars
    smma_5 = close_series.rolling(window=5, min_periods=5).mean()
    alligator_lips = smma_5.shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike (1.8x 20-period average) - balanced threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(alligator_jaw[i]) or np.isnan(alligator_teeth[i]) or 
            np.isnan(alligator_lips[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = bullish; Lips < Teeth < Jaw = bearish
            # Elder Ray: Bull Power > 0 and Bear Power < 0 for strong momentum
            # Volume confirmation
            if (alligator_lips[i] > alligator_teeth[i] > alligator_jaw[i] and 
                bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            elif (alligator_lips[i] < alligator_teeth[i] < alligator_jaw[i] and 
                  bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses (Lips < Teeth) or Elder Ray weakens
            if alligator_lips[i] < alligator_teeth[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Lips > Teeth) or Elder Ray weakens
            if alligator_lips[i] > alligator_teeth[i] or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals