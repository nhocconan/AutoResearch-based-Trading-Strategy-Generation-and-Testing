#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Hypothesis: Trade Camarilla pivot level bounces on 4h with 12h trend filter and volume confirmation.
# Uses 12h EMA25/50 for trend direction, Camarilla levels from 1d for support/resistance,
# and volume surge for confirmation. In bull markets, buy bounces off support with 12h uptrend;
# in bear markets, sell bounces off resistance with 12h downtrend. Designed for low trade frequency
# and robustness across regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h trend: EMA25/50 crossover
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d OHLC for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: H = high, L = low, C = close
    # Support 1: C - (H-L)*1.12/12
    # Support 2: C - (H-L)*1.12/6
    # Resistance 1: C + (H-L)*1.12/12
    # Resistance 2: C + (H-L)*1.12/6
    rng = high_1d - low_1d
    camarilla_s1 = close_1d - rng * 1.12 / 12
    camarilla_s2 = close_1d - rng * 1.12 / 6
    camarilla_r1 = close_1d + rng * 1.12 / 12
    camarilla_r2 = close_1d + rng * 1.12 / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 4h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema25_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below S2 OR stoploss hit
            if close[i] < camarilla_s2_aligned[i] or close[i] < camarilla_s1_aligned[i] - 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above R2 OR stoploss hit
            if close[i] > camarilla_r2_aligned[i] or close[i] > camarilla_r1_aligned[i] + 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price touches S1 with 12h uptrend and volume surge
            if (low[i] <= camarilla_s1_aligned[i] and  # Touches or pierces S1
                ema25_12h_aligned[i] > ema50_12h_aligned[i] and  # 12h uptrend
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches R1 with 12h downtrend and volume surge
            elif (high[i] >= camarilla_r1_aligned[i] and  # Touches or pierces R1
                  ema25_12h_aligned[i] < ema50_12h_aligned[i] and  # 12h downtrend
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals