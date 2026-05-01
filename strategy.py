#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes (overbought/oversold) with 1d Elder Ray confluence and volume confirmation
# Williams %R identifies momentum extremes; Elder Ray confirms bull/bear power alignment.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR14_Extreme_1dElderRay_Confluence_v1"
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
    
    # 1d HTF data for Elder Ray and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 6h volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 14 for Williams %R/EMA and 20 for volume MA, plus buffer
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Williams %R extremes: < -80 = oversold, > -20 = overbought
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        
        # Elder Ray conditions: bull power > 0 (bulls in control), bear power < 0 (bears in control)
        bulls_in_control = bull_power_aligned[i] > 0
        bears_in_control = bear_power_aligned[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold, bulls in control, volume spike
            if williams_oversold and bulls_in_control and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought, bears in control, volume spike
            elif williams_overbought and bears_in_control and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought or bears taking control
            if williams_r_aligned[i] > -20 or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold or bulls taking control
            if williams_r_aligned[i] < -80 or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals