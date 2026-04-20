#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA(50) trend + volume confirmation
# - Long when price breaks above 4h Donchian high(20) + price > 12h EMA(50) + volume > 1.5x 20-period average
# - Short when price breaks below 4h Donchian low(20) + price < 12h EMA(50) + volume > 1.5x 20-period average
# - Exit when price crosses back through Donchian midpoint or trend reverses
# - Uses 4h price channels for structure, 12h EMA for trend filter, volume for confirmation
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h timeframe
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA/Donchian/volume warmup
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_12h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 12h EMA + volume spike
            if price > donch_high[i] and price > ema_50_12h_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 12h EMA + volume spike
            elif price < donch_low[i] and price < ema_50_12h_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint or trend turns bearish
            if price < donch_mid[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint or trend turns bullish
            if price > donch_mid[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0