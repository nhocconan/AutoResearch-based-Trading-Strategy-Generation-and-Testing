#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray + volume confirmation
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Williams Alligator (jaw/teeth/lips) from 6h defines trend: lips > teeth > jaw = bullish, reverse = bearish
# 1d Elder Ray (Bull Power = high - EMA13, Bear Power = EMA13 - low) confirms trend strength
# Volume spike (>1.5 * 20-period EMA on 6h) ensures participation
# Discrete position sizing (0.25) minimizes fee churn
# Works in bull (Alligator aligned up + Bull Power > 0) and bear (Alligator aligned down + Bear Power > 0) markets

name = "6h_WilliamsAlligator_1dElderRay_Volume"
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
    
    # 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    # Williams Alligator: SMAs of median price (typical price)
    typical_price_6h = (df_6h['high'].values + df_6h['low'].values + df_6h['close'].values) / 3
    jaw = pd.Series(typical_price_6h).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(typical_price_6h).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(typical_price_6h).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # 1d data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = high - EMA13, Bear Power = EMA13 - low
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA (6h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_alligator and bull_power_aligned[i] > 0:
                # Long: Alligator bullish + Bull Power positive + volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_alligator and bear_power_aligned[i] > 0:
                # Short: Alligator bearish + Bear Power positive + volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop or weak power
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish or Bear Power becomes positive (trend weakening)
            if bearish_alligator or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish or Bull Power becomes positive (trend weakening)
            if bullish_alligator or bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals