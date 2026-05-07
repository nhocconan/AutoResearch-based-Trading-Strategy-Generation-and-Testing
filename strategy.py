#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray and volume confirmation.
# Uses 1w trend filter (EMA50) for long-term direction and 1d volume spike for entry timing.
# Williams Alligator (Jaw, Teeth, Lips) identifies trend presence and direction.
# Elder Ray (Bull Power/Bear Power) measures trend strength relative to EMA13.
# Designed to work in both bull and bear markets by following 1w trend.
# Target: 12-37 trades/year per symbol to avoid fee drag.
name = "12h_WilliamsAlligator_ElderRay_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w trend filter: 50-period EMA on close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # 1d volume average for spike detection
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=13, adjust=False, min_periods=13).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Williams Alligator (13,8,5 SMAs on median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Williams Alligator conditions: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power < 0 for strong trend
        strong_long = bull_power[i] > 0 and bear_power[i] < 0
        strong_short = bull_power[i] < 0 and bear_power[i] > 0
        
        # Volume spike: current volume > 1.5x 13-period EMA
        vol_spike = vol_avg_1d_aligned[i] > 0 and volume[i] / vol_avg_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long: Alligator alignment + strong Bull Power + volume spike in uptrend
            long_condition = alligator_long and strong_long and vol_spike and uptrend
            # Short: Alligator alignment + strong Bear Power + volume spike in downtrend
            short_condition = alligator_short and strong_short and vol_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator reverses or Elder Ray weakens
            if not (alligator_long and strong_long) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator reverses or Elder Ray weakens
            if not (alligator_short and strong_short) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals