#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1w EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA trend direction.
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) (using 13-period EMA on 6h).
- Trend filter: 1w EMA(34) slope (rising/falling) determines bias.
- Entry: Long when Bull Power > 0 AND 1w EMA rising AND volume > 1.5 * 20-period volume MA.
         Short when Bear Power < 0 AND 1w EMA falling AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit).
- Volume confirmation avoids weak breakouts.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull/bear: EMA trend filter ensures we only trade with the higher-timeframe trend.
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate EMA slope: rising if current > previous, falling if current < previous
    ema_slope = np.diff(ema_34_1w, prepend=ema_34_1w[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Align 1w EMA slope to 6h
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Elder Ray Index on 6h: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 34, 20)  # Need enough 1w bars for EMA and lookback for EMA13/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        vol_spike = volume_spike[i]
        ema_rise = ema_rising_aligned[i]
        ema_fall = ema_falling_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                if ema_rise and curr_bull > 0:  # Uptrend + bullish momentum
                    signals[i] = 0.25
                    position = 1
                elif ema_fall and curr_bear < 0:  # Downtrend + bearish momentum
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative (momentum fading)
            if curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive (momentum fading)
            if curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0