#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with daily RSI filter and volume confirmation.
# Long when: Price breaks above Donchian upper (20-period high), daily RSI > 50, volume > 1.5x 20-period average
# Short when: Price breaks below Donchian lower (20-period low), daily RSI < 50, volume > 1.5x 20-period average
# Exit when: Price crosses back through the 20-period midpoint
# Donchian provides clear breakout levels, daily RSI filters trend direction, volume confirms breakout strength.
# Target: 20-30 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "4h_Donchian20_RSI50_Volume_Filter"
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
    
    # 1-day data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 4H data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Calculate daily RSI(14) for trend filter
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align 1D RSI to 4H timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(mid_20[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        mid = mid_20[i]
        rsi = rsi_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian upper, RSI > 50, volume spike
            if (price > upper and close[i-1] <= upper and 
                rsi > 50 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian lower, RSI < 50, volume spike
            elif (price < lower and close[i-1] >= lower and 
                  rsi < 50 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals