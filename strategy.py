#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_donchian_breakout_v1
# Uses weekly Donchian breakout with volume confirmation and ATR stop on daily chart.
# In bull markets: buy breakouts above weekly high with volume.
# In bear markets: sell breakdowns below weekly low with volume.
# Weekly timeframe reduces noise, daily execution improves timing.
# Target: 20-40 trades per year (80-160 over 4 years) to minimize fee drag.
name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_20w = df_1w['high'].rolling(window=20, min_periods=20).max().values
    low_20w = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (wait for weekly close)
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20w)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Volume confirmation: volume > 2.0 * 50-period average on daily
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # ATR for dynamic sizing and stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if Donchian levels not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly Donchian high with volume
        if close[i] > donchian_high[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly Donchian low with volume
        elif close[i] < donchian_low[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout or ATR-based stop
        elif position == 1:
            # Exit long on breakdown below weekly low or 2*ATR stop
            if close[i] < donchian_low[i] or close[i] < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on breakout above weekly high or 2*ATR stop
            if close[i] > donchian_high[i] or close[i] > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals