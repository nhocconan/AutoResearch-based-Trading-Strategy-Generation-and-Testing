#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when: close > Donchian upper(20) and EMA(34) rising and volume > 1.5x 20-period average
# Short when: close < Donchian lower(20) and EMA(34) falling and volume > 1.5x 20-period average
# Exit when: price crosses back below Donchian midpoint (for long) or above midpoint (for short)
# Donchian channels provide clear breakout levels, EMA34 filters trend direction, volume confirms strength.
# Works in bull (buy breakouts) and bear (sell breakdowns). Target: 20-30 trades/year per symbol.
name = "4h_Donchian_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA34 on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema34 = ema34_1d_aligned[i]
        ema34_prev = ema34_1d_aligned[i-1]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        mid_val = donchian_mid[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: break above upper band, EMA rising, volume spike
            if (close[i] > high_20_val and ema34 > ema34_prev and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band, EMA falling, volume spike
            elif (close[i] < low_20_val and ema34 < ema34_prev and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if close[i] < mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if close[i] > mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals