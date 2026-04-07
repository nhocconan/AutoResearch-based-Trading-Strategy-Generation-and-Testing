#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with daily trend filter and volume confirmation
# Uses Donchian(20) breakout on 12h for entry, daily EMA(50) for trend filter,
# and volume > 1.5x 20-period average for confirmation. Includes ATR-based stoploss.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear markets via mean-reversion
# at extreme volatility spikes (filtered by daily trend to avoid counter-trend trades).

name = "12h_donchian20_daily_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: price breaks above Donchian high + daily uptrend + volume
        if (close[i] > highest_high[i] and 
            close[i] > ema_1d_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + daily downtrend + volume
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_1d_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        
        # Stoploss: close below/above entry ± 2*ATR
        elif i > 50:
            # Long exit
            if signals[i-1] > 0 and close[i] < close[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
            # Short exit
            elif signals[i-1] < 0 and close[i] > close[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = signals[i-1]
    
    return signals