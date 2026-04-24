#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + volume confirmation. Primary timeframe: 4h, HTF: 1d/1w for trend regime.
- Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend: MJAW > TEETH > LIPS = uptrend, reverse = downtrend.
- Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
- Entry: Long when Alligator uptrend + Bull Power > 0 + volume spike. Short when Alligator downtrend + Bear Power > 0 + volume spike.
- Exit: When Alligator trend reverses (JAW/TEETH/LIPS crossover) or volume dries up.
- Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h (JAW=13, TEETH=8, LIPS=5)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Elder Ray on 4h (EMA13 for power calculation)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals with volume spike and HTF trend filter
            if volume_spike[i]:
                # Bullish entry: Alligator uptrend + Bull Power positive + close > HTF EMA34
                if (jaw[i] > teeth[i] > lips[i]) and (bull_power[i] > 0) and (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Alligator downtrend + Bear Power positive + close < HTF EMA34
                elif (jaw[i] < teeth[i] < lips[i]) and (bear_power[i] > 0) and (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator trend reverses or volume dries up
            if not (jaw[i] > teeth[i] > lips[i]) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator trend reverses or volume dries up
            if not (jaw[i] < teeth[i] < lips[i]) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_ElderRay_VolumeSpike_1dEMA34_v1"
timeframe = "4h"
leverage = 1.0