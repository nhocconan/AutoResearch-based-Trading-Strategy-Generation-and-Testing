#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses daily EMA(34) for trend direction (from prior day's close) to filter breakouts.
# Enters long when price breaks above R1 in uptrend, short when breaks below S1 in downtrend.
# Volume spike (>1.5x 20-bar average) confirms breakout strength.
# Camarilla levels derived from prior day's OHLC provide institutional support/resistance.
# Designed for 4h timeframe to balance trade frequency and signal quality.
# Target: 20-50 trades/year per symbol to stay within fee limits (<200 total over 4 years).
# Works in both bull and bear markets via trend filter and breakout logic.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla levels and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each day
    # R4 = close + 1.5 * (high - low), R3 = close + 1.1 * (high - low)
    # R2 = close + 0.6 * (high - low), R1 = close + 0.318 * (high - low)
    # S1 = close - 0.318 * (high - low), S2 = close - 0.6 * (high - low)
    # S3 = close - 1.1 * (high - low), S4 = close - 1.5 * (high - low)
    rng = high_1d - low_1d
    r1 = close_1d + 0.318 * rng
    s1 = close_1d - 0.318 * rng
    
    # Align 1d indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in uptrend (close > EMA34) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend (close < EMA34) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                if (close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0