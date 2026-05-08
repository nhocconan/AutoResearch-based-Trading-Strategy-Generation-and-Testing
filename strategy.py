#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter, volume confirmation, and ATR-based volatility filter
# Works in bull (breakouts) and bear (mean reversion via volatility filter)
# Target: 20-50 trades/year to minimize drag
name = "4h_Donchian_Breakout_1dTrend_Volume_Volatility"
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
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high + uptrend (close > 1d EMA50) + volume + volatility filter
            long_cond = (close[i] > high_max[i]) and \
                        (close[i] > ema_50_1d_aligned[i]) and \
                        volume_confirmed[i] and \
                        (atr[i] > 0.5 * np.nanmedian(atr[max(0, i-50):i+1]))  # volatility not too low
            
            # Short: break below Donchian low + downtrend (close < 1d EMA50) + volume + volatility filter
            short_cond = (close[i] < low_min[i]) and \
                         (close[i] < ema_50_1d_aligned[i]) and \
                         volume_confirmed[i] and \
                         (atr[i] > 0.5 * np.nanmedian(atr[max(0, i-50):i+1]))
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low (mean reversion) or volatility collapse
            if close[i] < low_min[i] or atr[i] < 0.3 * np.nanmedian(atr[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high or volatility collapse
            if close[i] > high_max[i] or atr[i] < 0.3 * np.nanmedian(atr[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals