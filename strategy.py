#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator strategy with 1d trend filter and volume confirmation. 
# Uses Alligator jaws (13-period SMA), teeth (8-period SMA), lips (5-period SMA) to identify 
# trend strength and direction. Long when lips > teeth > jaws and price above 1d EMA50 with volume spike.
# Short when lips < teeth < jaws and price below 1d EMA50 with volume spike.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag in ranging/bear markets.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: > 1.8x 30-period average (moderate threshold to balance signals)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    # Williams Alligator components (12h timeframe)
    # Jaws: 13-period SMA, shifted 8 bars
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws, 8)
    jaws[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # 1d HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaws (Alligator bullish alignment) AND price > 1d EMA50 AND volume spike
            if (lips[i] > teeth[i] > jaws[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaws (Alligator bearish alignment) AND price < 1d EMA50 AND volume spike
            elif (lips[i] < teeth[i] < jaws[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator bearish alignment (Lips < Teeth < Jaws) OR price crosses below 1d EMA50
            if (lips[i] < teeth[i] < jaws[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator bullish alignment (Lips > Teeth > Jaws) OR price crosses above 1d EMA50
            if (lips[i] > teeth[i] > jaws[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals