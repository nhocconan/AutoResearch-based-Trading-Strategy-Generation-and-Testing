#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and volume confirmation
# Uses weekly Camarilla pivot levels (R4/S4) to determine long-term trend direction
# Only takes Donchian breakouts in the direction of weekly trend (R4 break for long, S4 break for short)
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by using weekly structure for trend determination
# Uses 1w for HTF regime and Donchian/volume for 6h entry timing

name = "6h_Donchian20_1wCamarillaRegime_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for Camarilla pivot regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (using typical price)
    # Typical price = (high + low + close) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    tp_h = typical_price.values
    tp_l = typical_price.values
    tp_c = typical_price.values
    
    # Weekly pivot point
    pp = (tp_h + tp_l + tp_c) / 3
    
    # Weekly Camarilla levels
    r4 = pp + ((tp_h - tp_l) * 1.1 / 2)
    r3 = pp + ((tp_h - tp_l) * 1.1 / 4)
    r2 = pp + ((tp_h - tp_l) * 1.1 / 6)
    r1 = pp + ((tp_h - tp_l) * 1.1 / 12)
    s1 = pp - ((tp_h - tp_l) * 1.1 / 12)
    s2 = pp - ((tp_h - tp_l) * 1.1 / 6)
    s3 = pp - ((tp_h - tp_l) * 1.1 / 4)
    s4 = pp - ((tp_h - tp_l) * 1.1 / 2)
    
    # Determine weekly trend: price above R4 = bullish, below S4 = bearish, between = neutral
    weekly_bullish = tp_c > r4
    weekly_bearish = tp_c < s4
    
    # Align weekly regime to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate 6x volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = lookback  # 20 bars for Donchian
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly regime
        bullish_regime = weekly_bullish_aligned[i] > 0.5
        bearish_regime = weekly_bearish_aligned[i] > 0.5
        
        if position == 0:  # Flat - look for new entries
            if bullish_regime:
                # In weekly bullish regime: only take long Donchian breakouts
                if close[i] > highest_high[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_regime:
                # In weekly bearish regime: only take short Donchian breakouts
                if close[i] < lowest_low[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral weekly regime: no trades (wait for clear direction)
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: weekly regime turns bearish OR price retouches midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if bearish_regime or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: weekly regime turns bullish OR price retouches midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if bullish_regime or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals