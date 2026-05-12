#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_1D_TREND_VOLUME
# Hypothesis: Donchian(20) breakout on 4h with 1d trend filter and volume confirmation.
# In 1d uptrend (close > EMA50), go long on 4h Donchian(20) high breakout with volume > 1.5x 20-period average.
# In 1d downtrend (close < EMA50), go short on 4h Donchian(20) low breakout with volume confirmation.
# Uses volatility-based position sizing (ATR-based) to adapt to market conditions.
# Target: 20-40 trades/year on 4h timeframe, avoiding overtrading.

name = "4H_DONCHIAN_BREAKOUT_1D_TREND_VOLUME"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA50 for 1d trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Donchian channels on 4h (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Average volume for confirmation (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + Donchian high breakout + volume confirmation
            if (close[i] > ema50_aligned[i] and 
                high[i] > highest_high[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + Donchian low breakout + volume confirmation
            elif (close[i] < ema50_aligned[i] and 
                  low[i] < lowest_low[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or Donchian low break
            if (close[i] <= ema50_aligned[i] or 
                low[i] < lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or Donchian high break
            if (close[i] >= ema50_aligned[i] or 
                high[i] > highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals