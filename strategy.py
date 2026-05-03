#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# Elder Ray (Bull/Bear Power) measures trend momentum relative to EMA13.
# Volume spike confirms conviction. Designed for 10-25 trades/year on 1d to minimize fee drag.
# Works in bull markets (Alligator aligned up, Elder Ray bullish) and bear markets (Aligator aligned down, Elder Ray bearish).

name = "1d_WilliamsAlligator_ElderRay_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate 1w EMA13 for HTF trend
    ema_13_1w = pd.Series(df_1w['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Calculate Elder Ray (Bull/Bear Power) using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period volume EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after sufficient warmup for Alligator
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_13_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_power_rising = bull_power[i] > 0 and (i == 13 or bull_power[i] > bull_power[i-1])
        bear_power_falling = bear_power[i] < 0 and (i == 13 or bear_power[i] < bear_power[i-1])
        
        if position == 0:
            # Long: Alligator aligned up + Bull Power rising + HTF uptrend + volume spike
            if alligator_long and bull_power_rising and ema_13_1w_aligned[i] > close[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + Bear Power falling + HTF downtrend + volume spike
            elif alligator_short and bear_power_falling and ema_13_1w_aligned[i] < close[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or Bull Power turns negative
            if not alligator_long or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or Bear Power turns positive
            if not alligator_short or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals