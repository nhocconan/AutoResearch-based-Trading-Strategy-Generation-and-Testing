#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear strength relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 AND volume spike
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 AND volume spike
# Uses discrete sizing 0.25 to limit fee drag; targets 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear via trend filter and measures underlying strength rather than price alone

name = "6h_ElderRay_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 6h EMA13 for Elder Ray (typical period)
    close_s = pd.Series(close)
    ema13_6h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13_6h  # Bull Power = High - EMA
    bear_power = low - ema13_6h   # Bear Power = Low - EMA
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema13_6h[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) AND Bear Power rising (less negative) 
            # AND uptrend (price > EMA34) AND volume spike
            if (bull_power[i] > 0 and bear_power[i] > bear_power[i-1] and 
                close[i] > ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling) AND Bull Power falling (less positive)
            # AND downtrend (price < EMA34) AND volume spike
            elif (bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and 
                  close[i] < ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative (buying pressure gone)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive (selling pressure gone)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals