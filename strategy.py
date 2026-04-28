#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray combination with 1w EMA34 trend filter.
# Uses 1d primary timeframe targeting 7-25 trades/year (30-100 total over 4 years).
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend presence and direction.
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength.
# 1w EMA34 provides higher timeframe trend alignment: only take longs when price > 1w EMA34,
# shorts when price < 1w EMA34 to avoid counter-trend trades.
# Volume confirmation (>1.5x 20-bar average) ensures breakout validity.
# Position size 0.25 balances return and drawdown control.
# Works in both bull and bear markets via trend filter + Alligator/Elder Ray logic.

name = "1d_WilliamsAlligator_ElderRay_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 13 or len(df_1w) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator: Smoothed Moving Average (SMA with specific periods)
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # 1w EMA34 for higher timeframe trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all HTF indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions:
        # Alligator sleeping (jaws, teeth, lips intertwined) = no trend
        # Alligator awakening (lines separated) = trend present
        # Direction: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_sleeping = (
            (lips_aligned[i] >= teeth_aligned[i] * 0.999) and (lips_aligned[i] <= teeth_aligned[i] * 1.001) and
            (teeth_aligned[i] >= jaw_aligned[i] * 0.999) and (teeth_aligned[i] <= jaw_aligned[i] * 1.001)
        )
        alligator_awake = not alligator_sleeping
        alligator_uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        alligator_downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Elder Ray conditions: measure trend strength
        strong_bull_power = bull_power_aligned[i] > 0
        strong_bear_power = bear_power_aligned[i] < 0
        
        # Higher timeframe trend filter: 1w EMA34
        price_above_1w_ema = close[i] > ema_34_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Long entry: Alligator uptrend + strong bull power + price above 1w EMA + volume spike
        long_entry = (
            alligator_awake and
            alligator_uptrend and
            strong_bull_power and
            price_above_1w_ema and
            vol_confirm
        )
        
        # Short entry: Alligator downtrend + strong bear power + price below 1w EMA + volume spike
        short_entry = (
            alligator_awake and
            alligator_downtrend and
            strong_bear_power and
            price_below_1w_ema and
            vol_confirm
        )
        
        # Exit conditions: Alligator sleeping or opposing Elder Ray signal
        long_exit = alligator_sleeping or (bear_power_aligned[i] > 0)
        short_exit = alligator_sleeping or (bull_power_aligned[i] < 0)
        
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