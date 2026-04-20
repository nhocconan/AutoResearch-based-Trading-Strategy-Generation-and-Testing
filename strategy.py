#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend + volume confirmation
# - Long when price breaks above 1d Donchian high(20) + price > 1w EMA(34) + volume > 2x 20-day average
# - Short when price breaks below 1d Donchian low(20) + price < 1w EMA(34) + volume > 2x 20-day average
# - Exit when price crosses back through 1d Donchian opposite bound or EMA trend reverses
# - Uses weekly EMA for trend filter and Donchian for breakout signals
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_min)
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate volume average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA/Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]  # Use 1d close for consistency
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above weekly EMA + volume spike
            if price > donchian_high[i] and price > ema_1w_aligned[i] and vol > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below weekly EMA + volume spike
            elif price < donchian_low[i] and price < ema_1w_aligned[i] and vol > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low or below weekly EMA
            if price < donchian_low[i] or price < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high or above weekly EMA
            if price > donchian_high[i] or price > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0