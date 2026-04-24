#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Williams %R calculation.
- Williams %R(14) calculated from 1d OHLC: values < -80 = oversold, > -20 = overbought.
- Entry: Long when Williams %R crosses above -80 from below with volume spike and close > 1d EMA34.
         Short when Williams %R crosses below -20 from above with volume spike and close < 1d EMA34.
- Exit: When Williams %R returns to the opposite extreme (-20 for longs, -80 for shorts) or opposing signal.
- Works in bull via buying oversold bounces in uptrend, in bear via selling overbought bounces in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for Williams %R and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams %R(14) for each 1d bar
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):  # Need 14 periods (0-13)
        highest_high = df_1d['high'].iloc[i-13:i+1].max()
        lowest_low = df_1d['low'].iloc[i-13:i+1].min()
        if highest_high == lowest_low:
            williams_r[i] = -50.0  # Avoid division by zero
        else:
            williams_r[i] = -100 * (highest_high - df_1d['close'].iloc[i]) / (highest_high - lowest_low)
    
    # Align 1d indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need enough 1d bars for EMA34 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for reversal signals with volume spike and trend filter
            if volume_spike[i] and i > 0:
                # Bullish reversal: Williams %R crosses above -80 from below
                if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                    close[i] > ema_34_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above
                elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                      close[i] < ema_34_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -20 (overbought) or bearish signal
            if williams_r_aligned[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -80 (oversold) or bullish signal
            if williams_r_aligned[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0