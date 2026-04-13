#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout with 12h volume regime filter and 1d trend confirmation
    # Long: price breaks above 6h Donchian(20) high AND 12h volume > 1.5x median AND price > 1d EMA200
    # Short: price breaks below 6h Donchian(20) low AND 12h volume > 1.5x median AND price < 1d EMA200
    # Exit: Donchian midpoint reversion (mean reversion in 6h timeframe)
    # Using 12h for volume regime (avoid low-volume false breakouts) and 1d for trend filter
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 12-37 trades/year (~50-150 over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume regime (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA200 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 6h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # 12h volume regime: >1.5x 50-period median (avoid low-volume breakouts)
    vol_12h = df_12h['volume'].values
    vol_median_12h = np.full(len(vol_12h), np.nan)
    for i in range(50, len(vol_12h)):
        vol_median_12h[i] = np.median(vol_12h[i-50:i])
    vol_median_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_12h)
    volume_regime = volume > (1.5 * vol_median_12h_aligned)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian midpoint for exit
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_regime[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime filter
        vol_ok = volume_regime[i]
        
        # Trend filter: only long if price > 1d EMA200, only short if price < 1d EMA200
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume regime + trend
        long_entry = (close[i] > donch_high[i]) and vol_ok and long_trend_ok
        short_entry = (close[i] < donch_low[i]) and vol_ok and short_trend_ok
        
        # Exit logic: return to Donchian midpoint (mean reversion)
        long_exit = close[i] < donch_mid[i]
        short_exit = close[i] > donch_mid[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0