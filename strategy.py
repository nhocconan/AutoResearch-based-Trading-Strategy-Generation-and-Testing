#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike
# - Williams Alligator (13,8,5 SMAs) defines trend: green (Lips>Teeth>Jaw) for long, red for short
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 confirms trend strength
# - Volume spike (>2.0x 20-period 12h average) ensures conviction
# - Only trade when Alligator is aligned (trending) and Elder Ray confirms direction
# - Weekly HTF (1w) provides major trend filter: only trade in direction of weekly close > open
# - Discrete position sizing ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Works in bull (Alligator green + Elder Ray bullish) and bear (Alligator red + Elder Ray bearish) markets
# - Weekly trend filter prevents trading against major trend, reducing false signals

name = "12h_1w_williams_alligator_elder_ray_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for major trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute weekly trend: close > open = bullish trend
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bullish = weekly_close > weekly_open  # True if weekly bullish
    
    # Align weekly trend to 12h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Load 1d data for Williams Alligator and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute Williams Alligator (SMAs: Jaw=13, Teeth=8, Lips=5)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (13-period SMA)
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMA)
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMA)
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Pre-compute Elder Ray (EMA13 for power calculation)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 12h volume SMA (20-period) for volume confirmation
    volume_12h = get_htf_data(prices, '12h')['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), volume_sma_20_12h)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Williams Alligator trend detection
        # Green (bullish): Lips > Teeth > Jaw
        # Red (bearish): Lips < Teeth < Jaw
        alligator_bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        alligator_bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Elder Ray confirmation
        # Bull Power > 0 indicates bulls in control
        # Bear Power < 0 indicates bears in control
        elder_ray_bullish = bull_power_aligned[i] > 0
        elder_ray_bearish = bear_power_aligned[i] < 0
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_filter_bullish = weekly_bullish_aligned[i] > 0.5
        weekly_filter_bearish = weekly_bullish_aligned[i] <= 0.5
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator green + Elder Ray bullish + volume confirmation + weekly bullish
        if alligator_bullish and elder_ray_bullish and vol_confirm and weekly_filter_bullish:
            enter_long = True
        
        # Short: Alligator red + Elder Ray bearish + volume confirmation + weekly bearish
        if alligator_bearish and elder_ray_bearish and vol_confirm and weekly_filter_bearish:
            enter_short = True
        
        # Exit conditions: opposite Alligator alignment or loss of Elder Ray confirmation
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns red OR Elder Ray turns bearish
            exit_long = (not alligator_bullish) or (not elder_ray_bullish)
        elif position == -1:
            # Exit short if Alligator turns green OR Elder Ray turns bullish
            exit_short = (not alligator_bearish) or (not elder_ray_bearish)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals