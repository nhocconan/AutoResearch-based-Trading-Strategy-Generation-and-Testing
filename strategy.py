#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_WilliamsAlligator_ElderRay_1d"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator (1d)
    # Jaws: 13-period SMMA of median price, shifted 8 bars forward
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    jaws_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws_raw, 8)
    jaws[:8] = np.nan
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Elder Ray Power (1d)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align indicators to 4h timeframe
    jaws_4h = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_4h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_4h = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_4h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_4h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 4h volume spike detection: current volume > 2.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or 
            np.isnan(bull_power_4h[i]) or np.isnan(bear_power_4h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Alligator bullish (jaws < teeth < lips) + Bull Power > 0 + volume spike
            alligator_bullish = (jaws_4h[i] < teeth_4h[i]) and (teeth_4h[i] < lips_4h[i])
            bull_power_positive = bull_power_4h[i] > 0
            
            if alligator_bullish and bull_power_positive and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            
            # Short entry: Alligator bearish (jaws > teeth > lips) + Bear Power < 0 + volume spike
            elif (jaws_4h[i] > teeth_4h[i]) and (teeth_4h[i] > lips_4h[i]) and (bear_power_4h[i] < 0) and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power turns negative
            alligator_bearish = (jaws_4h[i] > teeth_4h[i]) and (teeth_4h[i] > lips_4h[i])
            bull_power_negative = bull_power_4h[i] < 0
            
            if alligator_bearish or bull_power_negative:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power turns positive
            alligator_bullish = (jaws_4h[i] < teeth_4h[i]) and (teeth_4h[i] < lips_4h[i])
            bear_power_positive = bear_power_4h[i] > 0
            
            if alligator_bullish or bear_power_positive:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator trend alignment with Elder Ray power confirmation and volume spike on 4h timeframe.
# Williams Alligator (jaws/teeth/lips) identifies trend direction and alignment.
# Elder Ray measures bull/bear power relative to EMA13 for trend strength confirmation.
# Volume spike >2.5x 20-period average ensures institutional participation in breakouts.
# Works in bull markets (alligator bullish alignment) and bear markets (alligator bearish alignment).
# Target: 20-35 trades/year to minimize fee decay while capturing strong trends.