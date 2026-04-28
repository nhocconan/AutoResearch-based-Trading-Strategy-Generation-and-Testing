#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume spike.
# Uses proven Williams Alligator (jaw/teeth/lips) for trend direction and Elder Ray
# (bull/bear power) for momentum confirmation. Volume spike (>2.0x 20-bar average)
# confirms breakout strength. Works in both bull and bear via Alligator alignment
# (jaws < teeth < lips for uptrend, jaws > teeth > lips for downtrend).
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Primary timeframe: 12h, HTF: 1w for regime filter (optional).

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
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
    
    # Get 1d data for Williams Alligator and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    # We'll use EMA as proxy for SMMA with appropriate spans
    median_price = (high + low) / 2.0
    df_1d_median = pd.Series(median_price)
    
    # Jaw (13), Teeth (8), Lips (5) - using EMA as SMMA approximation
    jaw = df_1d_median.ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = df_1d_median.ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = df_1d_median.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = df_1d_median.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    bull_power = high - ema_13  # 1d values
    bear_power = low - ema_13   # 1d values
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike: >2.0x 20-bar average volume (stricter confirmation)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for volume MA and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend conditions
        # Uptrend: jaws < teeth < lips
        # Downtrend: jaws > teeth > lips
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        is_uptrend = (jaw_val < teeth_val) and (teeth_val < lips_val)
        is_downtrend = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Elder Ray momentum conditions
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        # Strong bullish momentum: bull power > 0 and increasing
        # Strong bearish momentum: bear power < 0 and decreasing
        # We'll use current values as proxy for momentum
        is_bullish_momentum = bull_val > 0
        is_bearish_momentum = bear_val < 0
        
        # Entry conditions with volume confirmation
        long_entry = is_uptrend and is_bullish_momentum and volume_spike[i]
        short_entry = is_downtrend and is_bearish_momentum and volume_spike[i]
        
        # Exit conditions: opposite Alligator alignment or loss of momentum
        long_exit = not is_uptrend or not is_bullish_momentum
        short_exit = not is_downtrend or not is_bearish_momentum
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals