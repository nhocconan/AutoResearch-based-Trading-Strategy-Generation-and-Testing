#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter
# Williams Alligator identifies trend presence (jaws/teeth/lips alignment) to avoid choppy markets
# Elder Ray measures bull/bear power via EMA13 relative to high/low for entry timing
# 1d EMA50 ensures alignment with higher timeframe trend for multi-timeframe confluence
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (bull power > 0 + alligator aligned up + price > 1d EMA50)
# Works in bear markets (bear power > 0 + alligator aligned down + price < 1d EMA50)

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator (6h timeframe)
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    # Using EMA as proxy for SMMA (standard practice) with proper alignment
    close_series = pd.Series(close)
    jaw = close_series.ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = close_series.ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = close_series.ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment
        # Uptrend: Lips > Teeth > Jaw (alligator mouth opening up)
        # Downtrend: Jaw > Teeth > Lips (alligator mouth opening down)
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_down = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull power positive + alligator aligned up + price above 1d EMA50
            if bull_power[i] > 0 and alligator_up and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive + alligator aligned down + price below 1d EMA50
            elif bear_power[i] > 0 and alligator_down and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear power becomes positive OR alligator alignment breaks down OR trend changes
            if bear_power[i] > 0 or not alligator_up or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull power becomes positive OR alligator alignment breaks up OR trend changes
            if bull_power[i] > 0 or not alligator_down or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals