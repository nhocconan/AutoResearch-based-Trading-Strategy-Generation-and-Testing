#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and ATR-based trailing stop
# - Uses 1d EMA(34) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 4h Donchian channel (20-period) for breakout entries
# - Requires volume > 1.3 * 20-period volume average for confirmation
# - ATR(14) trailing stop: exit long when price < highest_high - 3.0 * ATR
# - ATR(14) trailing stop: exit short when price > lowest_low + 3.0 * ATR
# - Fixed position size 0.25 to manage drawdown and reduce fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull markets via breakouts above resistance, in bear via breakdowns below support

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Pre-compute 4h Donchian channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high over past 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over past 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.3 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop
            if close[i] < highest_high_since_entry - 3.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop
            if close[i] > lowest_low_since_entry + 3.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries in direction of 1d trend with volume confirmation
            if uptrend and close[i] > donchian_high[i] and volume_confirm[i]:
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif downtrend and close[i] < donchian_low[i] and volume_confirm[i]:
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals