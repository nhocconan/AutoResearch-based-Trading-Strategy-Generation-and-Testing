#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams Alligator with 1d Elder Ray trend filter and volume confirmation
    # Williams Alligator (Jaw, Teeth, Lips) identifies trend direction and strength.
    # Elder Ray (Bull/Bear Power) from daily timeframe confirms institutional bias.
    # Volume spike validates participation. Designed to work in both bull and bear markets
    # by following the higher timeframe trend with strict entry conditions.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Williams Alligator (13,8,5 SMAs)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Williams Alligator components
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Load 1d data for Elder Ray (Bull Power, Bear Power)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
            # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
            lips_up = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
            lips_down = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
            bull_strong = bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1]
            bear_weak = bear_power_aligned[i] < 0 and bear_power_aligned[i] < bear_power_aligned[i-1]
            
            # Long: Bullish alignment + volume confirmation
            if lips_up and bull_strong and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume confirmation
            elif lips_down and bear_weak and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator reverses or Elder Ray divergence
            if position == 1:
                lips_down = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
                bear_strong = bear_power_aligned[i] < 0 and bear_power_aligned[i] < bear_power_aligned[i-1]
                if lips_down or bear_strong:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lips_up = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
                bull_strong = bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1]
                if lips_up or bull_strong:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dElderRay_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0