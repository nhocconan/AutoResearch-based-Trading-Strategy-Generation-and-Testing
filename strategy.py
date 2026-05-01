#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray with 1w trend filter for BTC/ETH.
# Uses Williams Alligator (jaw/teeth/lips) to detect trend absence/presence.
# Elder Ray (bull/bear power) measures trend strength via EMA13.
# 1w EMA34 trend filter ensures alignment with weekly structure.
# Discrete position sizing (0.25) balances return and drawdown. Target: 30-100 trades over 4 years.

name = "1d_WilliamsAlligator_ElderRay_1wEMA34_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator: SMA(13,8,5) shifted
    # Jaw: SMA(13,8), Teeth: SMA(8,5), Lips: SMA(5,3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: EMA13 for trend, Bull/Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Alligator (max shift 8) and EMA13
    start_idx = max(13, 8, 5, 13) + 8  # 21
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Rising Bull Power: current > previous
        # Falling Bear Power: current < previous (more negative)
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        bear_power_falling = i > 0 and bear_power[i] < bear_power[i-1]
        
        elder_long = bull_power[i] > 0 and bull_power_rising
        elder_short = bear_power[i] < 0 and bear_power_falling
        
        # 1w trend filter
        uptrend_1w = close[i] > ema_34_1w_aligned[i]
        downtrend_1w = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend AND Elder Ray bullish AND 1w uptrend
            if alligator_long and elder_long and uptrend_1w:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend AND Elder Ray bearish AND 1w downtrend
            elif alligator_short and elder_short and downtrend_1w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator downtrend OR Elder Ray bearish
            if not (alligator_long and elder_long):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator uptrend OR Elder Ray bullish
            if not (alligator_short and elder_short):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals