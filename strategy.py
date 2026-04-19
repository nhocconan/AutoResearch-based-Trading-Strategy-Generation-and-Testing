#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend filter + volume confirmation
# Bull Power = High - EMA13(Close), Bear Power = EMA13(Close) - Low
# Long when Bull Power > 0, Bear Power < 0, price > 1d EMA34, volume > 1.5x 1d avg volume
# Short when Bull Power < 0, Bear Power > 0, price < 1d EMA34, volume > 1.5x 1d avg volume
# Exit when Elder Ray signal reverses (Bull Power < 0 for long, Bear Power < 0 for short)
# Uses Elder Ray for momentum, 1d EMA for trend filter, volume for confirmation.
# Target: 15-25 trades/year per symbol.
name = "6h_ElderRay_EMA13_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 and average volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d average volume (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6-day EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, price > 1d EMA34, volume spike
            if bp > 0 and br < 0 and price > ema34 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Bull Power < 0, Bear Power > 0, price < 1d EMA34, volume spike
            elif bp < 0 and br > 0 and price < ema34 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power < 0 (momentum lost)
            if bp < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power < 0 (momentum lost)
            if br < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals