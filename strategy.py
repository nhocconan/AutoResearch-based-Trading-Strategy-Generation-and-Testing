#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with Donchian(20) breakout + volume confirmation + ATR filter
# Uses 1d HTF for trend alignment (price > 1d EMA50 for longs, < for shorts)
# Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull/bear markets
# Entry: 12h price breaks Donchian(20) with volume > 1.5x average and ATR > 0.6% of price
# Trend filter: 1d EMA50 direction (long only when price > EMA50, short only when price < EMA50)
# Exit: signal=0 when conditions not met or opposite signal triggers
# Position size: 0.25 (discrete to reduce churn)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above Donchian high with volume and trend filter
        if (close[i] > highest_20[i] and            # Breakout above 20-period high
            close[i] > ema_50_1d_aligned[i] and    # Above 1d EMA50 (uptrend filter)
            volume_ratio[i] > 1.5 and              # Strong volume confirmation
            atr_14[i] > 0.006 * close[i]):         # Adequate volatility
            signals[i] = 0.25
            
        # Short conditions: price breaks below Donchian low with volume and trend filter
        elif (close[i] < lowest_20[i] and          # Breakdown below 20-period low
              close[i] < ema_50_1d_aligned[i] and  # Below 1d EMA50 (downtrend filter)
              volume_ratio[i] > 1.5 and            # Strong volume confirmation
              atr_14[i] > 0.006 * close[i]):       # Adequate volatility
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_EMA50_Trend_Filter"
timeframe = "12h"
leverage = 1.0