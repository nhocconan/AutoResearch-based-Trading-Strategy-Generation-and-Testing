#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Williams Alligator components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, 3 bars ahead)
    lips_1w = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator to 12h
    jaw_12h = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_12h = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_12h = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Get 1d HTF data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray to 12h
    bull_power_12h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_12h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 12h TRIX (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(bull_power_12h[i]) or np.isnan(bear_power_12h[i]) or
            np.isnan(trix[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Alligator aligned (JAW > TEETH > LIPS) - bullish alignment
        # 2. Elder Ray Bull Power > 0 (buying pressure)
        # 3. TRIX turning up (positive slope) - momentum confirmation
        # 4. Volume confirmation: volume > 1.3x average
        if (jaw_12h[i] > teeth_12h[i] > lips_12h[i] and
            bull_power_12h[i] > 0 and
            trix[i] > trix[i-1] and
            volume_ratio[i] > 1.3):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Alligator inverted (JAW < TEETH < LIPS) - bearish alignment
        # 2. Elder Ray Bear Power < 0 (selling pressure)
        # 3. TRIX turning down (negative slope) - momentum confirmation
        # 4. Volume confirmation: volume > 1.3x average
        elif (jaw_12h[i] < teeth_12h[i] < lips_12h[i] and
              bear_power_12h[i] < 0 and
              trix[i] < trix[i-1] and
              volume_ratio[i] > 1.3):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Alligator_ElderRay_TRIX_Volume_Filter"
timeframe = "12h"
leverage = 1.0