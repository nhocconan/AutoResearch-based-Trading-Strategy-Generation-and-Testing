#!/usr/bin/env python3
# 6h_ElderRay_RayBand_Breakout
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# When Bull Power turns positive and Bear Power negative with EMA13 slope, it signals strong momentum.
# Combined with 1-week trend filter (EMA34) to avoid counter-trend trades. Works in bull markets by
# capturing strong up moves and in bear markets by catching strong down moves. Uses 6h timeframe for
# lower frequency to reduce fee drag.

name = "6h_ElderRay_RayBand_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34) and EMA13 (13)
    start_idx = max(34, 13)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Elder Ray signals: Bull Power > 0 and Bear Power < 0 with momentum
        bullish = bull_power[i] > 0 and bear_power[i] < 0
        bearish = bull_power[i] < 0 and bear_power[i] > 0
        
        if position == 0:
            # Long entry: weekly uptrend + Bull Power positive AND Bear Power negative
            if uptrend and bullish:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + Bear Power positive AND Bull Power negative
            elif downtrend and bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR Elder Ray turns bearish
            if not uptrend or not bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR Elder Ray turns bullish
            if not downtrend or not bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals