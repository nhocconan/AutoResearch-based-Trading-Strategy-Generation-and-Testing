#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combination with volume confirmation.
- Williams Alligator (JAW/TEETH/LIPS) identifies trend direction and strength
- Elder Ray (Bull/Bear Power) measures momentum behind price moves
- Volume spike (>2.0x average) confirms institutional participation
- 1w EMA50 ensures alignment with weekly trend (avoid counter-trend in major reversals)
- Discrete position size 0.25 to manage drawdown in volatile markets
- Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
- Designed for BTC/ETH performance in both bull (trend following) and bear (mean reversion via Alligator jaws)
"""

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
    
    # Williams Alligator: SMAs of median price with different periods
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Shift Alligator lines by respective periods to avoid look-ahead (SMAs use past data)
    jaw = np.roll(jaw, 8)   # 13-5=8 period shift for proper alignment
    teeth = np.roll(teeth, 5) # 8-3=5 period shift
    lips = np.roll(lips, 3)   # 5-2=3 period shift
    
    # Set initial values to NaN where data insufficient
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: > 2.0x 50-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # 1w data for EMA50 trend filter (weekly timeframe for strong trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50)  # volume MA, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Alligator alignment: jaws < teeth < lips = bearish, jaws > teeth > lips = bullish
        alligator_bullish = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_bearish = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        if position == 0:
            # Long: Alligator bullish AND Bull Power > 0 AND price above 1w EMA50 AND volume confirmation
            if alligator_bullish and bull_power[i] > 0 and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND price below 1w EMA50 AND volume confirmation
            elif alligator_bearish and bear_power[i] < 0 and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power <= 0 OR price crosses below 1w EMA50
            if not alligator_bullish or bull_power[i] <= 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power >= 0 OR price crosses above 1w EMA50
            if not alligator_bearish or bear_power[i] >= 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0