#!/usr/bin/env python3
name = "6h_Alligator_ElderRay_1dTrend"
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
    volume = prices['volume'].values
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Alligator: Jaw(13), Teeth(8), Lips(5) - all SMMA (using EMA as proxy)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = lips[i] > teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_elder = bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1])
        bear_elder = bear_power[i] < 0 and (i == start_idx or bear_power[i] < bear_power[i-1])
        
        if position == 0:
            # Long: Bullish Alligator + Bull Elder Ray + above 1d EMA50
            if bullish_alligator and bull_elder and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Bear Elder Ray + below 1d EMA50
            elif bearish_alligator and bear_elder and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Alligator reverses or Elder Ray weakens
            if position == 1:
                if not bullish_alligator or not bull_elder or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not bearish_alligator or not bear_elder or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals