#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d EMA Trend + Volume Confirmation
# - Long when price breaks above 4h Donchian upper (20) + price > 1d EMA50 + volume > 1.5x 20-period average
# - Short when price breaks below 4h Donchian lower (20) + price < 1d EMA50 + volume > 1.5x 20-period average
# - Exit when price crosses back through the Donchian midline (10-period average of high/low)
# - Trend filter uses daily EMA50 to avoid counter-trend trades
# - Volume filter ensures breakouts have conviction
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 19-50 trades per year per symbol (75-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 4h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper (20-period high)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midline (10-period average of high/low)
    donch_mid = (pd.Series(high).rolling(window=10, min_periods=10).mean() + 
                 pd.Series(low).rolling(window=10, min_periods=10).mean()) / 2
    donch_mid = donch_mid.values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + price > 1d EMA50 + volume > 1.5x average
            if price > donch_high[i] and price > ema_1d_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + price < 1d EMA50 + volume > 1.5x average
            elif price < donch_low[i] and price < ema_1d_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midline
            if price < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midline
            if price > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1dEMA_VolumeBreakout"
timeframe = "4h"
leverage = 1.0