#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# - Uses Williams Alligator (Jaw: SMA13, Teeth: SMA8, Lips: SMA5) on 4h for trend
# - 1d EMA34 as trend filter: only trade long when price > EMA34, short when price < EMA34
# - Volume confirmation: current volume > 1.5x 20-period average
# - Williams Alligator signals: Lips above Teeth above Jaw = bullish, reverse = bearish
# - Exit when Alligator lines re-converge (Lips crosses Teeth) or price crosses 8 SMA
# - Aims for 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Williams Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 4h: Jaw (SMA13), Teeth (SMA8), Lips (SMA5)
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        
        if position == 0:
            # Bullish Alligator: Lips > Teeth > Jaw + price > 1d EMA34 + volume surge
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and price > ema_34_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Bearish Alligator: Lips < Teeth < Jaw + price < 1d EMA34 + volume surge
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and price < ema_34_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: Lips crosses below Teeth OR price crosses below 8 SMA (Teeth)
            if lips[i] < teeth[i] or price < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Lips crosses above Teeth OR price crosses above 8 SMA (Teeth)
            if lips[i] > teeth[i] or price > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMAFilter_Volume"
timeframe = "4h"
leverage = 1.0