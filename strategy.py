#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray + volume spike confirmation.
- Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
- Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13
- Volume spike (>2.0x 20-period average) confirms institutional participation
- HTF 1d trend filter ensures alignment with daily momentum
- Designed for low-frequency, high-conviction trades on 12h timeframe
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
- Works in both bull (Alligator aligned up) and bear (Alligator aligned down) regimes
"""

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
    
    # Williams Alligator: SMAs of median price (typical price)
    median_price = (high + low + close) / 3.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Shift Alligator lines by their respective periods to avoid look-ahead
    jaw = np.roll(jaw, 13)
    teeth = np.roll(teeth, 8)
    lips = np.roll(lips, 5)
    jaw[:13] = np.nan
    teeth[:8] = np.nan
    lips[:5] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 13)  # volume MA, 1d EMA, EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Jaw > Teeth > Lips = bullish, Jaw < Teeth < Lips = bearish
        alligator_bull = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_bear = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish Alligator AND Bull Power > 0 AND price above 1d EMA34 AND volume confirmation
            if alligator_bull and bull_power[i] > 0 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND Bear Power < 0 AND price below 1d EMA34 AND volume confirmation
            elif alligator_bear and bear_power[i] < 0 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR price crosses below 1d EMA34
            if not alligator_bull or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR price crosses above 1d EMA34
            if not alligator_bear or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0