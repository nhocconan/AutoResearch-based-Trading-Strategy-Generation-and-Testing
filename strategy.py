#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter and volume spike confirmation. In trending markets (price > 1d EMA34), buy breaks above R1, sell breaks below S1. Uses volume > 1.5x 20-period average for confirmation. Discrete position sizing (0.25) minimizes fee churn. Designed for 50-150 trades over 4 years by requiring confluence of breakout, trend, and volume. Works in bull/bear via trend alignment: only long in uptrends, only short in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate Camarilla levels from previous 12h bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_multiplier = 1.1 / 12
    
    # Shift high/low/close by 1 to get previous bar levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar: use current values (no look-ahead)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    R1 = prev_close + camarilla_multiplier * (prev_high - prev_low)
    S1 = prev_close - camarilla_multiplier * (prev_high - prev_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        vol_spike = volume_spike[i]
        
        # Entry logic: only trade with trend and volume spike
        if vol_spike:
            if htf_trend[i] == 1 and close[i] > R1[i]:  # Long breakout in uptrend
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif htf_trend[i] == -1 and close[i] < S1[i]:  # Short breakdown in downtrend
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # No valid breakout - hold or exit
                if position == 1 and close[i] < prev_close[i]:  # Exit long on close below prev close
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > prev_close[i]:  # Exit short on close above prev close
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
        else:
            # No volume spike - hold or exit
            if position == 1 and close[i] < prev_close[i]:  # Exit long on close below prev close
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > prev_close[i]:  # Exit short on close above prev close
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0