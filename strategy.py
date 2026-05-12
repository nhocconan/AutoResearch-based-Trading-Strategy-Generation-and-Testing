#!/usr/bin/env python3
name = "12h_WilliamsAlligator_ElderRay_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR TREND FILTER (EMA13) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # === 1D DATA FOR WILLIAMS ALLIGATOR ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator lines (13, 8, 5 SMAs with 8, 5, 3 shifts)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips, additional_delay_bars=0)
    
    # Elder Ray Power (13-period EMA)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (high_1d - ema13_1d).values
    bear_power = (low_1d - ema13_1d).values
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power, additional_delay_bars=0)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power, additional_delay_bars=0)
    
    # === 12H VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_1w_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Williams Alligator alignment: all three lines ordered
        # Bullish: lips > teeth > jaw
        # Bearish: lips < teeth < jaw
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray: bull power > 0 and bear power < 0 for strong trend
        strong_bull = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        strong_bear = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        if position == 0:
            # LONG: Bullish Alligator + 1W uptrend + Elder Ray bull + volume spike
            if (bullish_alligator and 
                close[i] > ema13_1w_aligned[i] and
                strong_bull and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator + 1W downtrend + Elder Ray bear + volume spike
            elif (bearish_alligator and 
                  close[i] < ema13_1w_aligned[i] and
                  strong_bear and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Any condition breaks
            if not (bullish_alligator and 
                    close[i] > ema13_1w_aligned[i] and
                    strong_bull):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Any condition breaks
            if not (bearish_alligator and 
                    close[i] < ema13_1w_aligned[i] and
                    strong_bear):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals