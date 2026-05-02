#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1d EMA50 trend filter and volume confirmation
# Uses 6h primary timeframe for lower trade frequency (target: 50-150 total trades over 4 years)
# Williams Alligator (JAWS=13, TEETH=8, LIPS=5) identifies trend via smoothed SMAs
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength
# EMA50 trend filter from 1d ensures alignment with higher timeframe momentum
# Volume spike (2.0x 20-period average) confirms institutional participation
# Designed for both bull and bear markets: Alligator alignment defines trend, Elder Ray filters strength
# Target: 75-125 total trades over 4 years (19-31/year) - within proven winning range for 6h

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Alligator on 6h data (SMAs with smoothing)
    # JAWS: 13-period SMMA, TEETH: 8-period SMMA, LIPS: 5-period SMMA
    # Using EMA as approximation for SMMA with proper min_periods
    jaws = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate Elder Ray Power (using 13-period EMA as reference)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and HTF data)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: JAWS > TEETH > LIPS = uptrend, JAWS < TEETH < LIPS = downtrend
            # Elder Ray: Bull Power > 0 and rising, Bear Power < 0 and falling
            # Long: Uptrend alignment + Bull Power > 0 + price > EMA50 + volume spike
            if (jaws[i] > teeth[i] > lips[i] and 
                bull_power[i] > 0 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend alignment + Bear Power < 0 + price < EMA50 + volume spike
            elif (jaws[i] < teeth[i] < lips[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses (JAWS < TEETH) OR Elder Ray turns weak
            if jaws[i] < teeth[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses (JAWS > TEETH) OR Elder Ray turns weak
            if jaws[i] > teeth[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals