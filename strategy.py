# 6h_TripleBarrier_Breakout_12hTrend_Volume
# Hypothesis: 6h price breaking above 12h Donchian(20) high or below low with 12h EMA50 trend filter and volume spike (volume > 1.5x 20-period EMA).
# Works in bull markets by buying breakouts above resistance in uptrend, and in bear markets by selling breakdowns below support in downtrend.
# Target: 20-40 trades/year to avoid excessive fee drag.
name = "6h_TripleBarrier_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: max of last 20 highs
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: min of last 20 lows
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + 12h uptrend + volume spike
            if (price > donch_high_aligned[i] and price > ema_50_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + 12h downtrend + volume spike
            elif (price < donch_low_aligned[i] and price < ema_50_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or trend reverses
            if price < donch_low_aligned[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or trend reverses
            if price > donch_high_aligned[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals