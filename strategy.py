#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation.
# Long when: Close breaks above R1, EMA34 rising, volume > 1.5x 20-period average
# Short when: Close breaks below S1, EMA34 falling, volume > 1.5x 20-period average
# Exit when: Close crosses back below R1 (for long) or above S1 (for short)
# Camarilla levels provide precise support/resistance, EMA34 filters trend direction, volume confirms breakout strength.
# Works in bull (breakout long) and bear (breakdown short). Target: 12-37 trades/year per symbol.
name = "12h_Camarilla_EMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA34 for trend (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema34 = ema34_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla levels from previous 1d candle
        prev_1d_idx = i // 288  # 288 = 12h bars per day (24h * 2)
        if prev_1d_idx < 1:
            signals[i] = 0.0
            continue
        
        # Get previous day's OHLC from 1d data
        if prev_1d_idx >= len(df_1d):
            signals[i] = 0.0
            continue
        
        ph = df_1d['high'].iloc[prev_1d_idx - 1]
        pl = df_1d['low'].iloc[prev_1d_idx - 1]
        pc = df_1d['close'].iloc[prev_1d_idx - 1]
        
        # Calculate Camarilla levels
        range_ = ph - pl
        r1 = pc + (range_ * 1.1 / 12)
        s1 = pc - (range_ * 1.1 / 12)
        
        if position == 0:
            # Long entry: Close breaks above R1, EMA34 rising, volume spike
            if (close[i] > r1 and 
                ema34 > ema34_1d_aligned[i-1] and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Close breaks below S1, EMA34 falling, volume spike
            elif (close[i] < s1 and 
                  ema34 < ema34_1d_aligned[i-1] and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses back below R1
            if close[i] < r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses back above S1
            if close[i] > s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals