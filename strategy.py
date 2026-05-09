#!/usr/bin/env python3
# Hypothesis: 1h EMA crossover (12/26) with 4h EMA50 trend filter and volume confirmation
# Long when 1h EMA12 > EMA26, price > 4h EMA50, volume > 1.5x 20-period average
# Short when 1h EMA12 < EMA26, price < 4h EMA50, volume > 1.5x 20-period average
# Exit when EMA crossover reverses or volume condition fails
# Uses 4h trend to avoid counter-trend whipsaws, volume to confirm momentum
# Position size: 0.20 to limit drawdown and reduce frequency
# Target: 15-37 trades/year (~60-150 over 4 years) by requiring trend + momentum + volume confluence

name = "1h_EMA12_26_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA12 and EMA26 for crossover signal
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # 4h EMA50 for trend filter (loaded once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA26
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish EMA crossover + above 4h EMA50 + volume spike
            if (ema12[i] > ema26[i] and 
                close[i] > ema50_4h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: bearish EMA crossover + below 4h EMA50 + volume spike
            elif (ema12[i] < ema26[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: EMA crossover turns bearish OR price below 4h EMA50
            if (ema12[i] < ema26[i]) or (close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: EMA crossover turns bullish OR price above 4h EMA50
            if (ema12[i] > ema26[i]) or (close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals