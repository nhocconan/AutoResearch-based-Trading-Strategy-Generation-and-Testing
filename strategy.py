#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R with 1-day trend filter and volume confirmation.
Buy when Williams %R crosses above -80 (oversold) in an uptrend (price > 1-day EMA50) with volume spike.
Sell when Williams %R crosses below -20 (overbought) in a downtrend (price < 1-day EMA50) with volume spike.
Exit on opposite Williams %R cross or trend reversal.
Williams %R captures mean reversion in extremes; 1-day EMA50 filters higher timeframe trend.
Volume spike confirms institutional interest. Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following daily trend while using 4h Williams %R for entries.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14 periods)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Williams %R and volume MA
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below), price > 1-day EMA50, volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
                close[i] > ema50_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above), price < 1-day EMA50, volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
                  close[i] < ema50_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -20 OR trend turns down
                if (williams_r[i] < -20 and williams_r[i-1] >= -20) or \
                   (close[i] < ema50_1d_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -80 OR trend turns up
                if (williams_r[i] > -80 and williams_r[i-1] <= -80) or \
                   (close[i] > ema50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0