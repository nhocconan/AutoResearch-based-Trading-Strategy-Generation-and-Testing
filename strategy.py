#!/usr/bin/env python3
# 6h_ElderRay_Alligator_WeeklyTrend
# Hypothesis: Use Elder Ray (Bull/Bear Power) on 6h with Alligator (3 SMAs) and weekly trend filter (EMA34 on 1w).
# Elder Ray measures bull/bear power via EMA13; Alligator filters sideways markets; weekly trend ensures alignment with higher timeframe.
# Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets by avoiding whipsaws via trend and momentum filters.

name = "6h_ElderRay_Alligator_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment: market is trending when lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
    alligator_bull = lips > teeth
    alligator_bear = lips < teeth
    # Also require teeth > jaw for bull, teeth < jaw for bear to avoid chop
    alligator_bull = alligator_bull & (teeth > jaw)
    alligator_bear = alligator_bear & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(alligator_bull[i]) or np.isnan(alligator_bear[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Alligator bullish alignment, weekly trend up
            if bull_power[i] > 0 and alligator_bull[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Alligator bearish alignment, weekly trend down
            elif bear_power[i] > 0 and alligator_bear[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power > 0 or Alligator turns bearish
            if bear_power[i] > 0 or not alligator_bull[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power > 0 or Alligator turns bullish
            if bull_power[i] > 0 or not alligator_bear[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals