#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Channel breakout with 12-hour trend filter, volume confirmation, and ATR-based stoploss.
# Long when: Price breaks above Donchian(20) high, 12h EMA34 > 12h EMA89 (uptrend), volume > 1.5x 20-period average
# Short when: Price breaks below Donchian(20) low, 12h EMA34 < 12h EMA89 (downtrend), volume > 1.5x 20-period average
# Exit when: Price crosses back through Donchian(20) midline (mean of 20-period high/low)
# Donchian captures breakouts, 12h EMA crossover filters trend direction, volume confirms breakout strength.
# Designed for 4h timeframe with target of 20-40 trades/year per symbol to minimize fee drag.
name = "4h_Donchian_Trend_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian Channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Get 12h EMA data for trend filter (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = pd.Series(df_12h['close'].values).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (2-period ATR, 10-period average)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 89)  # Wait for Donchian and slow EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(ema89_12h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Donchian breakout + uptrend + volume spike
            if (close[i] > highest_20[i] and 
                ema34_12h_aligned[i] > ema89_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakdown + downtrend + volume spike
            elif (close[i] < lowest_20[i] and 
                  ema34_12h_aligned[i] < ema89_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian midline OR stoploss hit
            if close[i] < donchian_mid[i] or close[i] < (prices['close'].values[i-1] - 2.0 * atr[i] if i > 0 else -np.inf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Donchian midline OR stoploss hit
            if close[i] > donchian_mid[i] or close[i] > (prices['close'].values[i-1] + 2.0 * atr[i] if i > 0 else np.inf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals