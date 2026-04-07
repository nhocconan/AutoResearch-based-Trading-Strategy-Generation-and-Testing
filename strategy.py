#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets. The 12-hour EMA ensures trades align with higher timeframe trend.
# Volume confirmation filters low-participation false breakouts. Designed for low frequency in 4h timeframe.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Uses discrete position sizing (0.25) to minimize churn and transaction costs.

name = "4h_donchian20_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour EMA20 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema20_12h = close_12h.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate Donchian channel (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_max.values
    donchian_lower = low_min.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # 12-hour trend: close above/below 12h EMA20
        trend_up = close[i] > ema20_12h_aligned[i]
        trend_down = close[i] < ema20_12h_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if trend turns down or price breaks below Donchian lower
            if not trend_up or close[i] <= donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if trend turns up or price breaks above Donchian upper
            if not trend_down or close[i] >= donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: 12h uptrend + price breaks above Donchian upper + volume confirmation
            if trend_up and close[i] > donchian_upper[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: 12h downtrend + price breaks below Donchian lower + volume confirmation
            elif trend_down and close[i] < donchian_lower[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals