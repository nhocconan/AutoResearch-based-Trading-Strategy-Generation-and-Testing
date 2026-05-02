#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-based breakout with 4h trend and 1d volume regime filter
# Targets 15-35 trades per year (60-140 total over 4 years) to minimize fee drag
# Uses London/NY session (08-20 UTC) for higher quality breakouts
# 4h EMA50 trend filter ensures alignment with higher timeframe trend
# 1d volume regime filter (volume > 1.5x 20-day average) confirms institutional participation
# Discrete position sizing 0.20 to balance exposure and minimize fee churn
# Works in bull/bear: trend filter prevents counter-trend entries, volume regime avoids low-activity periods

name = "1h_SessionBreakout_4hEMA50_1dVolRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed for performance)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume regime: volume > 1.5x 20-day average
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20d = vol_1d.rolling(window=20, min_periods=20).mean().shift(1).values
    vol_regime_1d = df_1d['volume'].values > (vol_ma_20d * 1.5)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_1d)
    
    # Calculate 1h Donchian breakout levels (15-period)
    high_ma = pd.Series(high).rolling(window=15, min_periods=15).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=15, min_periods=15).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and session)
    start_idx = 15
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper AND above 4h EMA50 AND volume regime
            if (close[i] > high_ma[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_regime_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower AND below 4h EMA50 AND volume regime
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_regime_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR below 4h EMA50
            if (close[i] < low_ma[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR above 4h EMA50
            if (close[i] > high_ma[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals